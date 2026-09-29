import argparse
import os
import subprocess
import sys
from os import path

import imageio.v3 as iio
import numpy as np
import torch
import tqdm
from glob import glob
from torch.utils.data import DataLoader
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, StructuralSimilarityIndexMeasure, \
    PeakSignalNoiseRatio

from utils.score import *

# Metrics initialization
ssim_model = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
psnr_model = PeakSignalNoiseRatio(data_range=1.0).cuda()
lpips_model = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).cuda()


class MyDataset:
    def __init__(self, gt_paths, render_paths, mask_paths):
        self.gt_paths = gt_paths
        self.render_paths = render_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.gt_paths)

    def __getitem__(self, idx):
        im_gt = iio.imread(self.gt_paths[idx])
        im_gsbody = iio.imread(self.render_paths[idx])
        mask = iio.imread(self.mask_paths[idx])
        return im_gt, im_gsbody, mask, self.gt_paths[idx]


def evaluate_folder(data_dir):
    # Paths to the files
    filenames = os.listdir(path.join(data_dir, 'gt'))
    filenames.sort()
    gt_paths = [path.join(data_dir, f'gt/{filename}') for filename in filenames]
    mask_paths = [path.join(data_dir, f'mask/{filename}') for filename in filenames]

    render_names = os.listdir(path.join(data_dir, 'result'))
    render_names.sort()
    render_paths = [path.join(data_dir, f'result/{filename}') for filename in render_names]

    # Dataset creation
    dataset = MyDataset(gt_paths, render_paths, mask_paths)
    dataloader = DataLoader(dataset=dataset, batch_size=None, num_workers=8, collate_fn=lambda x: x)

    # Metrics list initialisation
    lpips_list = []
    ssim_list = []
    psnr_list = []


    # Image processing
    for i, data in tqdm.tqdm(enumerate(dataloader)):
        im_gt, im_gsbody, mask, datapath = data
        assert len(mask.shape) == 2
        mask = mask > 128

        im_gt_crop, im_gsbody_crop = crop_image(mask, 512, im_gt, im_gsbody)

        im_gt = torch.tensor(im_gt / 255).permute(2, 0, 1).float().cuda()[None]
        im_gsbody = torch.tensor(im_gsbody / 255).permute(2, 0, 1).float().cuda()[None]

        im_gt_crop = torch.tensor(im_gt_crop / 255).permute(2, 0, 1).float().cuda()[None]
        im_gsbody_crop = torch.tensor(im_gsbody_crop / 255).permute(2, 0, 1).float().cuda()[None]

        psnr_value = psnr_model(im_gsbody, im_gt).item()
        ssim_value = ssim_model(im_gsbody, im_gt).item()
        lpips_value = lpips_model(im_gsbody_crop, im_gt_crop).item()

        psnr_list.append(psnr_value)
        ssim_list.append(ssim_value)
        lpips_list.append(lpips_value)

    # Averaging metrics
    psnr_list = np.array(psnr_list)
    ssim_list = np.array(ssim_list)
    lpips_list = np.array(lpips_list)

    psnr_val = psnr_list.mean()
    ssim_val = ssim_list.mean()
    lpips_val = lpips_list.mean()

    result = subprocess.run(
        [sys.executable, '-m', 'pytorch_fid', path.join(data_dir, 'gt'), path.join(data_dir, 'result')],
        capture_output=True, text=True
    )
    output = result.stdout.strip() or result.stderr.strip()
    if not output:
        raise RuntimeError('pytorch_fid produced no output')
    fid_val = float(output.split()[-1])

    return psnr_val, ssim_val, lpips_val, fid_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate image quality metrics')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to directory containing gt, render and mask folders')
    args = parser.parse_args()

    test_folders = glob(os.path.join(args.data_dir, '*'))
    test_folders = sorted(test_folders)
    lpips_list = []
    ssim_list = []
    psnr_list = []
    fid_list = []

    for folder in test_folders:
        print("Eval" + folder)
        psnr_val, ssim_val, lpips_val, fid_val = evaluate_folder(folder)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        lpips_list.append(lpips_val)
        fid_list.append(fid_val)

        print(f"PSNR: {psnr_val}")
        print(f"SSIM: {ssim_val}")
        print(f"LPIPS: {lpips_val}")
        print(f"FID: {fid_val}")
        print("*" * 10)

    # Averaging metrics
    psnr_list = np.array(psnr_list)
    ssim_list = np.array(ssim_list)
    lpips_list = np.array(lpips_list)
    fid_list = np.array(fid_list)

    psnr_val = psnr_list.mean()
    ssim_val = ssim_list.mean()
    lpips_val = lpips_list.mean()
    fid_val = fid_list.mean()

    print(f"Total PSNR: {psnr_val}")
    print(f"Total SSIM: {ssim_val}")
    print(f"Total LPIPS: {lpips_val}")
    print(f"Total FID: {fid_val}")
