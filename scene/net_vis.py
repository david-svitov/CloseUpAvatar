
IMAGE_ENCODE = 'torch'

import os
from os import path
import time
import torch
import sys
from tqdm import tqdm
import numpy as np
import json
import pickle
from scipy.spatial.transform import Rotation

from utils.image_utils import encode_bytes
from utils.net_utils import net_init, recv, send, is_connected
from scene.gaussian_model import GaussianModel

def load_pose_list(file_path):
    with open(file_path, 'r') as file:
        pose_list = json.load(file)
    for pose in pose_list:
        for key in ['pose', 'Rh', 'Th']:
            pose[key] = np.array(pose[key]).astype(np.float32)
    return pose_list

def load_cam_list(file_path):
    with open(file_path, 'r') as file:
        cam_list = json.load(file)
    for cam in cam_list:
        for key in ['K', 'w2c']:
            cam[key] = np.array(cam[key]).astype(np.float32)
    return cam_list

def load_model(model_dir):
    ckpt_path = path.join(model_dir, sorted([pth for pth in os.listdir(model_dir) if 'chkpnt' in pth and '.pth' in pth], key=lambda s: (len(s), s))[-1])
    gaussians = GaussianModel()
    load_data = torch.load(ckpt_path, weights_only=False)
    first_iter = load_data['iteration']
    print(f'Loading checkpoint from ITER {first_iter}')
    gaussians.restore(load_data)
    return gaussians

class Visualizer:
    gaussians: GaussianModel
    cam_list = None
    pose_list = None
    is_send_initial_data = False
    def __init__(self, in_training=False):
        self.in_training = in_training
    
    def net_init(self, ip, port):
        net_init(ip, port)

    def load_model(self, model_dir):
        self.gaussians = load_model(model_dir)
        self.load_cams_poses(model_dir)

    def load_cams_poses(self, model_dir):
        self.cam_list = load_cam_list(path.join(model_dir, 'cameras.json'))
        self.pose_list = load_pose_list(path.join(model_dir, 'poses.json'))

    @staticmethod
    def pklbytes_to_data(pklbytes):
        if pklbytes is None: return None 
        data = pickle.loads(pklbytes)
        data['cam_pos'] = np.linalg.inv(data['w2c'])[:3,3]

        cuda_keys = ['K', 'w2c', 'background', 'cam_pos']
        cpu_keys = ['pose', 'Rh', 'Th']
        for key in cuda_keys:
            data[key] = torch.from_numpy(data[key]).float().cuda(non_blocking=True)
        for key in cpu_keys:
            data[key] = torch.from_numpy(data[key]).float()
        return data

    @torch.no_grad()
    def visualizing(self):
        if not is_connected(): return False

        data = self.pklbytes_to_data(recv())
        if data is None: return False

        gaussians = self.gaussians   
        if not self.in_training and 'is_test' in data:
            gaussians.is_test = data['is_test']
        if 'is_gsparam_bs' in data:
            gaussians.is_gsparam_bs = data['is_gsparam_bs']
        if 'is_dxyz_bs' in data:
            gaussians.is_dxyz_bs = data['is_dxyz_bs']
        if 'sh_degree' in data:
            sh_degree = max(0, min(data['sh_degree'], 1))
            gaussians.sh_degree = sh_degree

        gaussians.Rh, gaussians.Th = data['Rh'], data['Th']
        gaussians.smpl_poses = data['pose']

        image, _, _ = gaussians.render(
            cam=data,
            scaling_modifier=data['scaling_modifier'], 
            background=data['background'],
        )
        image = (torch.clamp(image, min=0, max=1.0) * 255).byte()
        if IMAGE_ENCODE != 'torch': image = image.contiguous().cpu().numpy()
        image_bytes = encode_bytes(image, IMAGE_ENCODE)
        
        ret_data = {}

        ret_data['image_bytes'] = image_bytes
        ret_data['gaussian_num'] = len(gaussians._xyz)
        
        if self.is_send_initial_data:
            self.is_send_initial_data = False
            ret_data['camera_list'] = self.cam_list
            ret_data['pose_list'] = self.pose_list

        send(pickle.dumps(ret_data))
        return True

