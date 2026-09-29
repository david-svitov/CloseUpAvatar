import os
import random
import sys
import numpy as np
from os import path
import cv2 as cv
from scipy.spatial.transform import Rotation
import skimage.morphology as sm
import tqdm
import json
import csv

import torch
import imageio.v3 as iio

stm = None


def data_to_cam(data: dict, non_blocking=True):
    img_list = ['image', 'mask', 'mask_boundary', 'pad_mask']
    tensor_list = ['K', 'w2c']
    const_list = ['height', 'width', 'frame_id', 'cam_id', 'idx']
    cpu_list = ['pose', 'beta', 'Rh', 'Th']
    global stm
    if stm is None: stm = torch.cuda.Stream()
    for k, v in data.items():
        if not (isinstance(v, torch.Tensor) or isinstance(v, np.ndarray)): continue

        if k in tensor_list:
            data[k] = torch.as_tensor(v).squeeze().cuda(non_blocking=non_blocking)
        elif k in cpu_list:
            data[k] = torch.as_tensor(v).squeeze()
        elif k in const_list:
            data[k] = v.squeeze().item()
        elif k in img_list:
            with torch.cuda.stream(stm):
                data[k] = torch.as_tensor(v).squeeze().cuda(non_blocking=non_blocking)
    return data


def get_dataset_type(datadir):
    if path.exists(path.join(datadir, 'calibration.json')):
        return ThumanDataset
    if path.exists(path.join(datadir, 'calibration_full.json')):
        return AVRexDataset
    if path.exists(path.join(datadir, 'calibration.csv')):
        return ActorsHQDataset
    raise RuntimeError


def get_scene_scale(cams):
    camtoworlds = np.stack([cam['w2c'] for cam in cams], axis=0)
    camera_locations = camtoworlds[:, :3, 3]
    scene_center = np.mean(camera_locations, axis=0)
    dists = np.linalg.norm(camera_locations - scene_center, axis=1)
    scene_scale = np.max(dists)
    return scene_scale


def apply_distortion(im_list, K, D):
    if np.sum(np.abs(D)) < 1e-4:
        return im_list
    for i, im in enumerate(im_list):
        im_list[i] = cv.undistort(im, K, D[None, ...])
    return im_list


def resize_image(image, mask, K, image_scaling=1):
    if image_scaling == 1: return image, mask, K

    H, W = int(image.shape[0] * image_scaling), int(image.shape[1] * image_scaling)
    image = cv.resize(image, (W, H), interpolation=cv.INTER_AREA)
    mask = cv.resize(mask, (W, H), interpolation=cv.INTER_NEAREST)
    K = np.copy(K)
    K[:2] = K[:2] * image_scaling

    return image, mask, K


def resize_and_crop_image(image, mask, K, image_scaling=1, crop_scale=None, x_offset=0, y_offset=0):
    if image_scaling == 1: return image, mask, K

    H, W = int(image.shape[0] * image_scaling), int(image.shape[1] * image_scaling)

    # Select random crop from the image
    if crop_scale is None:
        crop_scale = random.uniform(image_scaling, 1.0)
    H_crop = int(image.shape[0] * crop_scale)
    W_crop = int(image.shape[1] * crop_scale)
    y_start = (image.shape[0] - H_crop) // 2
    x_start = (image.shape[1] - W_crop) // 2

    y_start = y_start + y_offset
    x_start = x_start + x_offset

    image = image[y_start:y_start + H_crop, x_start:x_start + W_crop]
    mask = mask[y_start:y_start + H_crop, x_start:x_start + W_crop]

    image = cv.resize(image, (W, H), interpolation=cv.INTER_AREA)
    mask = cv.resize(mask, (W, H), interpolation=cv.INTER_NEAREST)
    K = np.copy(K)
    scale_factor = image_scaling / crop_scale
    K[0, 0] *= scale_factor  # fx
    K[1, 1] *= scale_factor  # fy
    K[0, 2] = (K[0, 2] - x_start) * scale_factor  # cx
    K[1, 2] = (K[1, 2] - y_start) * scale_factor  # cy

    return image, mask, K

def resize_and_pad_image(image, mask, K, image_scaling=1, max_scaling=2, pad_scale=None):
    if pad_scale is None:
        pad_scale = random.uniform(1.0, max_scaling)
    if pad_scale == 1.0:
        pad_mask = np.ones_like(mask) * 255
        return image, mask, pad_mask, K

    H, W = int(image.shape[0] * image_scaling), int(image.shape[1] * image_scaling)
    H_target = int(image.shape[0] * pad_scale)
    W_target = int(image.shape[1] * pad_scale)
    pad_h = H_target - image.shape[0]
    pad_w = W_target - image.shape[1]

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    image = cv.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right,
                              cv.BORDER_CONSTANT, value=0)
    pad_mask = cv.copyMakeBorder(np.ones_like(mask) * 255, pad_top, pad_bottom, pad_left, pad_right,
                                 cv.BORDER_CONSTANT, value=0)
    mask = cv.copyMakeBorder(mask, pad_top, pad_bottom, pad_left, pad_right,
                             cv.BORDER_CONSTANT, value=0)

    image = cv.resize(image, (W, H), interpolation=cv.INTER_AREA)
    mask = cv.resize(mask, (W, H), interpolation=cv.INTER_NEAREST)
    pad_mask = cv.resize(pad_mask, (W, H), interpolation=cv.INTER_NEAREST)

    K = np.copy(K)
    scale_factor = image_scaling / pad_scale
    K[0, 0] *= scale_factor  # fx
    K[1, 1] *= scale_factor  # fy
    K[0, 2] = (K[0, 2] + pad_left) * scale_factor  # cx
    K[1, 2] = (K[1, 2] + pad_top) * scale_factor  # cy

    return image, mask, pad_mask, K


def get_mask_boundary(mask, kernel_size):
    mask_ero = sm.binary_erosion(mask, sm.disk(kernel_size))
    mask_dil = sm.binary_dilation(mask, sm.disk(kernel_size))
    mask_boundary = mask_ero ^ mask_dil
    return mask_boundary


class AVRexDataset:
    def __init__(self, datadir, frame_ids, cam_ids, background=np.zeros(3, dtype=np.float32),
                 image_scaling=1, is_in_memory=False):
        indices = []

        annots = AVRexDataset.load_cams_data(datadir)

        for frame_id in frame_ids:
            for cam_id in cam_ids:
                cam_name, img_name = annots[cam_id]['name'], f'{frame_id:08d}.jpg'
                if path.exists(path.join(datadir, f'{cam_name}/{img_name}')) and \
                        path.exists(path.join(datadir, f'{cam_name}/mask/pha/{img_name}')):
                    indices.append((frame_id, cam_id))

        self.indices = indices
        self.image_scaling = image_scaling
        self.datadir = datadir
        self.annots = annots
        self.is_in_memory = is_in_memory
        self.background = background
        self.is_load_image = True

        if is_in_memory:
            self.data_list = []
            self.is_in_memory = False
            for idx in tqdm.tqdm(range(len(self.indices))):
                self.data_list.append(self.__getitem__(idx))
            self.is_in_memory = True
        else:
            cv.setNumThreads(0)

        self.smpl_params = AVRexDataset.load_pose_data(datadir)

    @staticmethod
    def load_cams_data(datadir, cam_file_name='calibration_full.json'):
        with open(path.join(datadir, cam_file_name), 'r') as file:
            data = json.load(file)

        cams = []
        for k, v in data.items():
            cam = {}
            cam['name'] = k
            cam['K'] = np.array(v['K']).astype(np.float32).reshape(3, 3)
            cam['R'] = np.array(v['R']).astype(np.float32).reshape(3, 3)
            cam['T'] = np.array(v['T']).astype(np.float32)
            cam['D'] = np.array(v['distCoeff']).astype(np.float32)
            T, R = cam['T'], cam['R']
            w2c = np.block([[R, T[:, None]], [0, 0, 0, 1]])
            cam['w2c'] = w2c
            cams.append(cam)
        return cams

    @staticmethod
    def load_pose_data(datadir):
        smpl_params = np.load(path.join(datadir, 'smpl_params.npz'), allow_pickle=True)
        smpl_params = dict(smpl_params)

        N_frame = len(smpl_params['global_orient'])
        beta = smpl_params['betas'][0]

        pose_list, Th_list, Rh_list = [], [], []
        for frame_id in range(N_frame):
            pose = np.concatenate([smpl_params['global_orient'][frame_id],
                                   smpl_params['body_pose'][frame_id],
                                   torch.zeros(3).float(),
                                   torch.zeros(6).float(),
                                   smpl_params['left_hand_pose'][frame_id],
                                   smpl_params['right_hand_pose'][frame_id], ], axis=0)
            Th = smpl_params['transl'][frame_id]
            Rh = np.eye(3, dtype=np.float32)

            pose_list.append(pose)
            Th_list.append(Th)
            Rh_list.append(Rh)

        pose_data = dict(pose=np.array(pose_list).astype(np.float32), Th=np.array(Th_list).astype(np.float32),
                         Rh=np.array(Rh_list).astype(np.float32), beta=beta.astype(np.float32))
        return pose_data

    @staticmethod
    def get_scene_scale(datadir, cam_file_name='calibration_full.json'):
        cams = AVRexDataset.load_cams_data(datadir, cam_file_name)
        return get_scene_scale(cams)

    @staticmethod
    def load_image_mask(datadir, cam_name, frame_id):
        image_path = path.join(datadir, f'{cam_name}/{frame_id:08d}.jpg')
        image = iio.imread(image_path)[..., :3]
        mask_path = path.join(datadir, f'{cam_name}/mask/pha/{frame_id:08d}.jpg')
        mask = iio.imread(mask_path)  # 1C u8
        return image, mask

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.is_in_memory: return self.data_list[idx]

        frame_id, cam_id = self.indices[idx]

        pose, Rh, Th, beta = self.smpl_params['pose'][frame_id], self.smpl_params['Rh'][frame_id], \
            self.smpl_params['Th'][frame_id], self.smpl_params['beta']

        # Load camera
        K, D, w2c = self.annots[cam_id]['K'], self.annots[cam_id]['D'], self.annots[cam_id]['w2c']
        # T: 3  R: 33  K: 33  D: 5
        cam_name = self.annots[cam_id]['name']
        # avatarrex camera: W2C  down y

        if self.is_load_image:
            image, mask = AVRexDataset.load_image_mask(self.datadir, cam_name, frame_id)
            image, mask = apply_distortion([image, mask], K, D)
            image, mask, K = resize_image(image, mask, K, self.image_scaling)

            image = image.astype(np.float32) / 255
            mask = mask[:, :, 0] > 128 if len(mask.shape) == 3 else mask > 128

            mask_boundary = get_mask_boundary(mask, 5)
        else:
            image = np.zeros((100, 100, 3), dtype=np.float32)
            mask = np.zeros((100, 100), dtype=bool)
            mask_boundary = mask

        data = {
            'K': torch.from_numpy(K).float(),
            'w2c': torch.from_numpy(w2c).float(),
            'image': torch.from_numpy(image).float(),
            'mask': torch.from_numpy(mask),
            'mask_boundary': torch.from_numpy(mask_boundary),
            'pad_mask': torch.ones_like(torch.from_numpy(mask)),
            'pose': torch.from_numpy(pose).float(),
            'Rh': torch.from_numpy(Rh).float(),
            'Th': torch.from_numpy(Th).float(),
            'beta': torch.from_numpy(beta).float(),
            'height': image.shape[0],
            'width': image.shape[1],
            'frame_id': frame_id,
            'cam_id': cam_id,
            'idx': idx,
        }

        return data


class ThumanDataset:
    def __init__(self, datadir, frame_ids, cam_ids, background=np.zeros(3, dtype=np.float32),
                 split='train', image_scaling=1, is_in_memory=False, random_rescaling=False,
                 zoom_in=False, zoom_out=False, x_offset=0, y_offset=0):

        indices = []
        annots = AVRexDataset.load_cams_data(datadir, cam_file_name='calibration.json')

        for frame_id in frame_ids:
            for cam_id in cam_ids:
                cam_name, img_name = annots[cam_id]['name'], f'{frame_id:08d}.jpg'
                if path.exists(path.join(datadir, f'images/{cam_name}/{img_name}')) and \
                        path.exists(path.join(datadir, f'masks/{cam_name}/{img_name}')):
                    indices.append((frame_id, cam_id))

        self.indices = indices
        self.image_scaling = image_scaling
        self.split = split
        self.datadir = datadir
        self.annots = annots
        self.is_in_memory = is_in_memory
        self.background = background
        self.is_load_image = True

        if is_in_memory:
            self.data_list = []
            self.is_in_memory = False
            for idx in tqdm.tqdm(range(len(self.indices))):
                self.data_list.append(self.__getitem__(idx))
            self.is_in_memory = True
        else:
            cv.setNumThreads(0)

        # load poses
        self.smpl_params = AVRexDataset.load_pose_data(datadir)

    @staticmethod
    def get_scene_scale(datadir):
        return AVRexDataset.get_scene_scale(datadir, cam_file_name='calibration.json')

    @staticmethod
    def load_image_mask(datadir, cam_name, frame_id):
        image_path = path.join(datadir, f'images/{cam_name}/{frame_id:08d}.jpg')
        image = iio.imread(image_path)[..., :3]
        mask_path = path.join(datadir, f'masks/{cam_name}/{frame_id:08d}.jpg')
        mask = iio.imread(mask_path)  # 1C u8
        return image, mask

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.is_in_memory: return self.data_list[idx]

        frame_id, cam_id = self.indices[idx]

        pose, Rh, Th, beta = self.smpl_params['pose'][frame_id], self.smpl_params['Rh'][frame_id], \
            self.smpl_params['Th'][frame_id], self.smpl_params['beta']

        # Load camera
        K, D, w2c = self.annots[cam_id]['K'], self.annots[cam_id]['D'], self.annots[cam_id]['w2c']
        # T: 3  R: 33  K: 33  D: 5
        cam_name = self.annots[cam_id]['name']
        # avatarrex camera: W2C  down y

        if self.is_load_image:
            image, mask = ThumanDataset.load_image_mask(self.datadir, cam_name, frame_id)
            image, mask = apply_distortion([image, mask], K, D)
            image, mask, K = resize_image(image, mask, K, self.image_scaling)

            image = image.astype(np.float32) / 255
            mask = mask[:, :, 0] > 128 if len(mask.shape) == 3 else mask > 128

            mask_boundary = get_mask_boundary(mask, 3)
        else:
            image = np.zeros((100, 100, 3), dtype=np.float32)
            mask = np.zeros((100, 100), dtype=bool)
            mask_boundary = mask

        data = {
            'K': torch.from_numpy(K).float(),
            'w2c': torch.from_numpy(w2c).float(),
            'image': torch.from_numpy(image).float(),
            'mask': torch.from_numpy(mask),
            'mask_boundary': torch.from_numpy(mask_boundary),
            'pad_mask': torch.ones_like(torch.from_numpy(mask)),
            'pose': torch.from_numpy(pose).float(),
            'Rh': torch.from_numpy(Rh).float(),
            'Th': torch.from_numpy(Th).float(),
            'beta': torch.from_numpy(beta).float(),
            'height': image.shape[0],
            'width': image.shape[1],
            'frame_id': frame_id,
            'cam_id': cam_id,
            'idx': idx,
        }

        return data


class ActorsHQDataset:
    def __init__(self, datadir, frame_ids, cam_ids, background=np.zeros(3, dtype=np.float32),
                 split='train', image_scaling=1, max_upscaling=2, is_in_memory=False, random_rescaling=True,
                 zoom_in=False, zoom_out=False, x_offset=0, y_offset=0):

        annots = ActorsHQDataset.load_cams_data(datadir)
        self.smpl_params = AVRexDataset.load_pose_data(datadir)

        # filter out repeated frames
        new_frame_ids = [frame_ids[0]]
        for i in range(1, len(frame_ids)):
            pose0 = self.smpl_params['pose'][new_frame_ids[-1]][3:22 * 3]
            pose1 = self.smpl_params['pose'][frame_ids[i]][3:22 * 3]
            if np.abs(pose0 - pose1).mean() > 1.5e-2:
                new_frame_ids.append(frame_ids[i])

        frame_ids = new_frame_ids

        # cannot model hair dynamics in actor06, actor04
        if 'actor06' in datadir:
            frame_ids = [f for f in frame_ids if f < 1400 or f > 1530]
        if 'actor04' in datadir:
            frame_ids = [f for f in frame_ids if f < 1380 or f > 1590]

        print('dataset frame num', len(frame_ids))

        indices = []
        print(np.min(frame_ids), np.max(frame_ids))
        for frame_id in frame_ids:
            for cam_id in cam_ids:
                cam_name = annots[cam_id]['name']
                img_name = f'{cam_name}_rgb{frame_id:06d}.jpg'
                msk_name = f'{cam_name}_mask{frame_id:06d}.png'
                if path.exists(path.join(datadir, f'rgbs/{cam_name}/{img_name}')) and \
                        path.exists(path.join(datadir, f'masks/{cam_name}/{msk_name}')):
                    indices.append((frame_id, cam_id))

        self.indices = indices
        self.image_scaling = image_scaling
        self.max_upscaling = max_upscaling
        self.split = split
        self.datadir = datadir
        self.annots = annots
        self.is_in_memory = is_in_memory
        self.background = background
        self.is_load_image = True
        self.random_rescaling = random_rescaling
        self.zoom_in = zoom_in
        self.zoom_out = zoom_out
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.prev_id = 0

        if is_in_memory:
            self.data_list = []
            self.is_in_memory = False
            for idx in tqdm.tqdm(range(len(self.indices))):
                self.data_list.append(self.__getitem__(idx))
            self.is_in_memory = True
        else:
            cv.setNumThreads(0)

    @staticmethod
    def load_cams_data(datadir):
        # code from AnimatableGaussians
        cams = []
        with open(path.join(datadir, 'calibration.csv'), 'r', newline='', encoding='utf-8') as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                cam_name = row['name']

                extr_mat = np.identity(4, np.float32)
                extr_mat[:3, :3] = \
                cv.Rodrigues(np.array([float(row['rx']), float(row['ry']), float(row['rz'])], np.float32))[0]
                extr_mat[:3, 3] = np.array([float(row['tx']), float(row['ty']), float(row['tz'])])
                extr_mat = np.linalg.inv(extr_mat)

                intr_mat = np.identity(3, np.float32)
                intr_mat[0, 0] = float(row['fx']) * float(row['w'])
                intr_mat[0, 2] = float(row['px']) * float(row['w'])
                intr_mat[1, 1] = float(row['fy']) * float(row['h'])
                intr_mat[1, 2] = float(row['py']) * float(row['h'])

                data = dict(name=cam_name, K=intr_mat, w2c=extr_mat, D=np.zeros(5, dtype=np.float32))
                cams.append(data)
        return cams

    @staticmethod
    def get_scene_scale(datadir):
        cams = ActorsHQDataset.load_cams_data(datadir)
        return get_scene_scale(cams)

    @staticmethod
    def load_image_mask(datadir, cam_name, frame_id):
        img_name = f'{cam_name}_rgb{frame_id:06d}.jpg'
        msk_name = f'{cam_name}_mask{frame_id:06d}.png'
        image_path = path.join(datadir, f'rgbs/{cam_name}/{img_name}')
        image = iio.imread(image_path)[..., :3]
        mask_path = path.join(datadir, f'masks/{cam_name}/{msk_name}')
        mask = iio.imread(mask_path)
        return image, mask

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.is_in_memory: return self.data_list[idx]

        frame_id, cam_id = self.indices[idx]

        pose, Rh, Th, beta = self.smpl_params['pose'][frame_id], self.smpl_params['Rh'][frame_id], \
            self.smpl_params['Th'][frame_id], self.smpl_params['beta']

        # Load camera
        K, D, w2c = self.annots[cam_id]['K'], self.annots[cam_id]['D'], self.annots[cam_id]['w2c']
        cam_name = self.annots[cam_id]['name']

        zero_pad_mask = False
        if self.is_load_image:
            try:
                image, mask = ActorsHQDataset.load_image_mask(self.datadir, cam_name, frame_id)
            except OSError:
                image, mask = ActorsHQDataset.load_image_mask(self.datadir, cam_name, self.prev_id)
                image = np.zeros_like(image)
                mask = np.zeros_like(mask)
                zero_pad_mask = True

            image, mask = apply_distortion([image, mask], K, D)
            if self.random_rescaling:
                if random.uniform(0, 1) < 0.5:
                    image, mask, K = resize_and_crop_image(image, mask, K, self.image_scaling)
                    pad_mask = np.ones_like(mask) * 255
                else:
                    image, mask, pad_mask, K = resize_and_pad_image(image, mask, K, self.image_scaling, self.max_upscaling)
            elif self.zoom_in:
                image, mask, K = resize_and_crop_image(image, mask, K, self.image_scaling, crop_scale=0.25,
                                                       x_offset=self.x_offset, y_offset=self.y_offset)
                pad_mask = np.ones_like(mask) * 255
            elif self.zoom_out:
                image, mask, pad_mask, K = resize_and_pad_image(image, mask, K, self.image_scaling, self.max_upscaling, pad_scale=2.0)
            else:
                image, mask, K = resize_image(image, mask, K, self.image_scaling)
                pad_mask = np.ones_like(mask) * 255

            image = image.astype(np.float32) / 255
            mask = mask[:, :, 0] > 128 if len(mask.shape) == 3 else mask > 128
            pad_mask = pad_mask[:, :, 0] > 128 if len(pad_mask.shape) == 3 else pad_mask > 128

            mask_boundary = get_mask_boundary(mask, 4)
        else:
            image = np.zeros((100, 100, 3), dtype=np.float32)
            mask = np.zeros((100, 100), dtype=bool)
            mask_boundary = mask
            pad_mask = np.ones_like(mask)

        self.prev_id = frame_id

        if zero_pad_mask:
            pad_mask = np.zeros_like(mask)

        data = {
            'K': torch.from_numpy(K).float(),
            'w2c': torch.from_numpy(w2c).float(),
            'image': torch.from_numpy(image).float(),
            'mask': torch.from_numpy(mask),
            'mask_boundary': torch.from_numpy(mask_boundary),
            'pad_mask': torch.from_numpy(pad_mask),
            'pose': torch.from_numpy(pose).float(),
            'Rh': torch.from_numpy(Rh).float(),
            'Th': torch.from_numpy(Th).float(),
            'beta': torch.from_numpy(beta).float(),
            'height': image.shape[0],
            'width': image.shape[1],
            'frame_id': frame_id,
            'cam_id': cam_id,
            'idx': idx,
        }

        return data