
import os
from os import path
import torch
from omegaconf import OmegaConf
import sys
from tqdm import tqdm
import numpy as np
import random
import pickle
import copy
import json
from argparse import ArgumentParser
import copy
from scipy.spatial.transform import Rotation
import imageio.v3 as iio
from torch.utils.data import DataLoader

from scene.dataset import get_dataset_type, data_to_cam
from scene.gaussian_model import GaussianModel
from scene.net_vis import load_model
from utils.config_utils import Config
from utils.image_utils import encode_bytes
from utils.smpl_utils import init_smpl_pose

def fovx_to_intrinsic(fovx, H, W):
    focal = W / 2 / np.tan(fovx/2)
    K = np.zeros((3, 3))
    K[0, 0] = focal
    K[1, 1] = focal
    K[2, 2] = 1
    K[0, 2], K[1, 2] = W/2, H/2
    return K.astype(np.float32)

def load_amass_pose_list(pose_path):
    data = np.load(pose_path)
    pose_list = []
    poses = data['poses'].astype(np.float32)
    trans = data['trans'].astype(np.float32)
    N = len(poses)

    # AMASS poses are noisy
    OPTIMIZE_AMASS = True
    if OPTIMIZE_AMASS:
        foo = poses[:,3:]
        foo[:, 13 * 3 + 2] -= 0.25
        foo[:, 12 * 3 + 2] += 0.25
        foo[:, 19 * 3: 20 * 3] = 0.
        foo[:, 20 * 3: 21 * 3] = 0.
        foo[:, 14 * 3] = 0.

        poses[:,3:] = foo

        # smooth
        win_size = 1
        poses_clone = np.copy(poses)
        trans_clone = np.copy(trans)
        frame_num = poses_clone.shape[0]
        poses[win_size: frame_num-win_size] = 0
        trans[win_size: frame_num-win_size] = 0
        for i in range(-win_size, win_size + 1):
            poses[win_size: frame_num-win_size] += poses_clone[win_size+i: frame_num-win_size+i]
            trans[win_size: frame_num-win_size] += trans_clone[win_size+i: frame_num-win_size+i]
        poses[win_size: frame_num-win_size] /= (2 * win_size + 1)
        trans[win_size: frame_num-win_size] /= (2 * win_size + 1)

    for i in range(N):
        pose_list.append(dict(pose=poses[i], Th=trans[i], Rh=np.eye(3, dtype=np.float32)))

    return pose_list

def load_thuman_pose_list(pose_path):
    smpl_params = np.load(pose_path, allow_pickle=True)
    smpl_params = dict(smpl_params)

    pose_list = []
    N = len(smpl_params['global_orient'])
    for frame_id in range(N):
        pose = np.concatenate([smpl_params['global_orient'][frame_id],
                    smpl_params['body_pose'][frame_id],
                    np.zeros(3,dtype=np.float32),
                    np.zeros(6,dtype=np.float32),
                    smpl_params['left_hand_pose'][frame_id],
                    smpl_params['right_hand_pose'][frame_id],], axis=0)
        Th = smpl_params['transl'][frame_id]
        Rh = np.eye(3, dtype=np.float32)
        pose_list.append(dict(pose=pose, Th=Th, Rh=Rh))
    return pose_list


def interpolate_cameras(w2c1, w2c2, t):
    new_w2c = (1 - t) * w2c1 + t * w2c2
    return new_w2c

def testing_novel_cam_pose_speed(gaussians: GaussianModel, out_dir, frame_ids, pose_list, cam, background):

    # warm up
    pose = pose_list[0]
    gaussians.smpl_poses = torch.as_tensor(pose['pose'])
    gaussians.Th = torch.as_tensor(pose['Th'])
    gaussians.Rh = torch.as_tensor(pose['Rh'])
    image, alpha, info = gaussians.render(cam, background=background)
    torch.cuda.synchronize()

    run_time = 0
    for frame_id in frame_ids:
        pose = pose_list[frame_id]
        gaussians.smpl_poses = torch.as_tensor(pose['pose'])
        gaussians.Th = torch.as_tensor(pose['Th'])
        gaussians.Rh = torch.as_tensor(pose['Rh'])

        image, alpha, info = gaussians.render(cam, background=background)
        image = (torch.clamp(image, min=0, max=1.0) * 255).byte().contiguous()
        run_time += info["run_time"]

    fps = len(frame_ids) / run_time * 1000
    print('FPS:', fps)

def testing_novel_cam_pose(gaussians: GaussianModel, out_dir, frame_ids, pose_list, cam, background, zero_translation):

    os.makedirs(path.join(out_dir), exist_ok=True)
    for frame_id in tqdm(frame_ids):
        pose = pose_list[frame_id]
        pose = copy.deepcopy(pose)

        gaussians.smpl_poses = torch.as_tensor(pose['pose']).cpu()
        if zero_translation:
            gaussians.Th = torch.zeros_like(torch.as_tensor(pose['Th']).cpu())
        else:
            gaussians.Th = torch.clone(torch.as_tensor(pose['Th']).cpu())
        gaussians.Rh = torch.as_tensor(pose['Rh']).cpu()
        image, alpha, info = gaussians.render(cam, background=background)

        image = (torch.clamp(image, min=0, max=1.0) * 255).byte().contiguous().cpu().numpy()
        iio.imwrite(path.join(out_dir, f'{frame_id:08d}.png'), image)

def testing_camera_move(gaussians: GaussianModel, out_dir, frame_ids, pose_list, cam, cam_new, background, zero_translation):

    os.makedirs(path.join(out_dir), exist_ok=True)
    frame_id = frame_ids[0]

    pose = pose_list[frame_id]
    pose = copy.deepcopy(pose)

    gaussians.smpl_poses = torch.as_tensor(pose['pose']).cpu()
    if zero_translation:
        gaussians.Th = torch.zeros_like(torch.as_tensor(pose['Th']).cpu())
    else:
        gaussians.Th = torch.clone(torch.as_tensor(pose['Th']).cpu())
    gaussians.Rh = torch.as_tensor(pose['Rh']).cpu()

    i = 0
    for t in tqdm(np.arange(0, 1, 0.01)):
        inter_cam = copy.deepcopy(cam)
        inter_cam['w2c'] = interpolate_cameras(cam['w2c'], cam_new['w2c'], t)
        image, alpha, info = gaussians.render(inter_cam, background=background)

        image = (torch.clamp(image, min=0, max=1.0) * 255).byte().contiguous().cpu().numpy()
        iio.imwrite(path.join(out_dir, f'{i:08d}.png'), image)
        i += 1


def testing_dataset(gaussians: GaussianModel, out_dir, dataset, background):
    test_dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    for k in ['gt', 'result', 'mask']:
        os.makedirs(path.join(out_dir, k), exist_ok=True)

    fps_list = []

    for cam in tqdm(test_dataloader):
        cam = data_to_cam(cam, non_blocking=False)
        frame_id = cam['frame_id']
        gaussians.smpl_poses = cam['pose']
        gaussians.Th, gaussians.Rh = cam['Th'], cam['Rh']

        image, alpha, info = gaussians.render(cam, background=background)

        run_time = info["run_time"]
        fps = 1 / run_time * 1000
        fps_list.append(fps)

        image = (torch.clamp(image, min=0, max=1.0) * 255).byte().contiguous().cpu().numpy()

        image_gt = cam['image']
        image_gt[~cam['mask']] = background
        image_gt = (image_gt * 255).byte().contiguous().cpu().numpy()
        mask = cam['mask'].byte().contiguous().cpu().numpy() * 255

        iio.imwrite(path.join(out_dir, f'gt/{frame_id:08d}.png'), image_gt)
        iio.imwrite(path.join(out_dir, f'result/{frame_id:08d}.png'), image)
        iio.imwrite(path.join(out_dir, f'mask/{frame_id:08d}.png'), mask)

    print("FPS:", np.array(fps_list[3:]).mean())


@torch.no_grad()
def testing(args: Config):
    init_smpl_pose()

    gaussians = load_model(args.model_dir)
    gaussians.is_test = args.test.is_test
    gaussians.prepare_test()
    background = torch.as_tensor(np.array(args.background)).float().cuda()

    # Dataset
    test_frame_ids = np.arange(args.test.begin_ith_frame, args.test.begin_ith_frame+args.test.num_frame, args.test.frame_interval).tolist()
    test_cam_ids = np.array(args.test.cam_ids).tolist()

    if args.test.cam_path is not None and args.test.pose_path is not None:
        with open(args.test.cam_path, 'r') as file:
            cam = json.load(file)
        cam['w2c'] = torch.as_tensor(np.array(cam['w2c']).reshape(4,4)).float().cuda()
        K = fovx_to_intrinsic(cam['fovx'] / 180 * np.pi, cam['height'], cam['width'])
        cam['K'] = torch.as_tensor(K).cuda()

        if 'smpl_params.npz' in args.test.pose_path:
            pose_list = load_thuman_pose_list(args.test.pose_path)
        else:
            pose_list = load_amass_pose_list(args.test.pose_path)

        if args.test.test_speed:
            testing_novel_cam_pose_speed(gaussians, args.out_dir, test_frame_ids, pose_list, cam, background)
        elif args.test.cam_path_move is not None:
            with open(args.test.cam_path_move, 'r') as file:
                cam_new = json.load(file)
            cam_new['w2c'] = torch.as_tensor(np.array(cam_new['w2c']).reshape(4, 4)).float().cuda()
            K = fovx_to_intrinsic(cam_new['fovx'] / 180 * np.pi, cam_new['height'], cam_new['width'])
            cam_new['K'] = torch.as_tensor(K).cuda()

            testing_camera_move(gaussians, args.out_dir, test_frame_ids, pose_list, cam, cam_new, background, args.test.zero_translation)
        else:
            testing_novel_cam_pose(gaussians, args.out_dir, test_frame_ids, pose_list, cam, background, args.test.zero_translation)
    else:
        DatasetType = get_dataset_type(args.data_dir)
        testset = DatasetType(
            datadir=args.data_dir,
            frame_ids=test_frame_ids,
            cam_ids=test_cam_ids,
            background=np.array(args.background),
            image_scaling=args.image_scaling,
            random_rescaling=args.test.random_rescaling,
            zoom_in=args.test.zoom_in,
            zoom_out=args.test.zoom_out,
            x_offset=args.test.x_offset,
            y_offset=args.test.y_offset,
        )

        testing_dataset(gaussians, args.out_dir, testset, background)

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing")

    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--model_dir', type=str, default='')
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='')

    parser.add_argument('--cam_path', type=str, default=None)
    parser.add_argument('--cam_path_move', type=str, default=None)
    parser.add_argument('--pose_path', type=str, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_speed', action='store_true')
    parser.add_argument("--zero_translation", action='store_true')
    pargs = parser.parse_args(sys.argv[1:])

    args = OmegaConf.load(pargs.config)
    args.data_dir, args.out_dir, args.model_dir, args.test.cam_path, args.test.pose_path, args.test.cam_path_move = (
        pargs.data_dir, pargs.out_dir, pargs.model_dir, pargs.cam_path, pargs.pose_path, pargs.cam_path_move)
    args.test.is_test, args.test.test_speed = pargs.test, pargs.test_speed
    args.test.zero_translation = pargs.zero_translation
    torch.backends.cuda.matmul.allow_tf32 = True

    testing(args)
