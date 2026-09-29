#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import sys
from argparse import ArgumentParser
from os import path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from scene.dataset import data_to_cam
from scene.gaussian_model import GaussianModel
from scene.net_vis import Visualizer
from scene.scene import Scene
from utils.config_utils import Config
from utils.general_utils import safe_state
from utils.image_utils import crop_image
from utils.loss_utils import l1_loss, mse_loss, ms_ssim, psnr, lpips_loss, dxyz_smooth_loss, gaussian_scaling_loss


def tensor_to_image(image):
    image = image.detach().cpu().numpy()
    image = np.clip(image, 0, 1)
    image = np.clip(np.rint(image * 255), 0, 255).astype(np.uint8)
    return image


def training(args: Config):
    gaussians = GaussianModel()
    scene = Scene(args, gaussians)
    gaussians.training_setup(args, scene.scene_scale)

    visualizer = Visualizer(in_training=True)
    visualizer.net_init(args.ip, args.port)
    visualizer.gaussians = gaussians
    visualizer.load_cams_poses(args.out_dir)

    background = torch.as_tensor(args.background).float().cuda()

    ema_vis_loss, ema_lpips_loss = 0.0, 0.0
    first_iter = 0
    progress_bar = tqdm(range(0, args.iterations), initial=first_iter, desc="TP")
    first_iter += 1
    trainloader_iter = iter(scene.trainloader)

    for iteration in range(first_iter, args.iterations + 1):
        if iteration % 30 == 0:
            visualizer.is_send_initial_data = True
            visualizer.visualizing()

        if iteration % (scene.trainloader_len - 1) == 0:
            trainloader_iter = iter(scene.trainloader)

        try:
            cam = next(trainloader_iter)
        except StopIteration:
            trainloader_iter = iter(scene.trainloader)
            cam = next(trainloader_iter)

        cam = data_to_cam(cam)
        bg = torch.rand(3, device="cuda") if args.random_background else background

        gaussians.smpl_poses = cam['pose']
        gaussians.Th, gaussians.Rh = cam['Th'], cam['Rh']

        image, alpha, info = gaussians.render(cam, background=bg)
        image = torch.clamp(image, 0, 1)
        image_gt, mask, mask_boundary, pad_mask = cam['image'], cam['mask'], cam['mask_boundary'], cam['pad_mask']
        image_gt[~mask] = bg
        image_gt[mask_boundary] = bg
        image[mask_boundary] = bg
        image[~pad_mask] = bg

        if iteration % 100 == 0 or iteration == first_iter:
            visualize = cv2.vconcat([
                tensor_to_image(image),
                tensor_to_image(image_gt)
            ])
            cv2.imwrite("visalize_dbg.png", visualize[..., ::-1])

        if iteration < 7000:
            pixel_loss = l1_loss(image, image_gt)
        else:
            pixel_loss = mse_loss(image, image_gt)
        msssimloss = ms_ssim(image, image_gt)
        dxyzsmoothloss = dxyz_smooth_loss(gaussians) * args.lambda_dxyz_smooth

        random_patch_flag = False if iteration < args.iteration_lpips_random_patch else True
        image_crop, image_gt_crop = crop_image(bg, mask, 512, random_patch_flag, image.permute(2, 0, 1),
                                               image_gt.permute(2, 0, 1))
        if iteration > args.iteration_lpips:
            lpipsloss = lpips_loss(image_crop.permute(1, 2, 0), image_gt_crop.permute(1, 2, 0)) * args.lambda_lpips
        else:
            lpipsloss = torch.tensor(0)

        # regularization
        lambda_normal = args.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = args.lambda_dist if iteration > 3000 else 0.0

        rend_dist = info["render_dist"]
        rend_normal  = info['render_normal']
        surf_normal = info['surf_normal']
        normal_error = 1 - (rend_normal * surf_normal)
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        impact = info["impact"]
        visibility_filter = impact > 0
        scaling_loss = args.lambda_scaling * gaussian_scaling_loss(gaussians.get_cano_scaling, visibility_filter, args.scaling_threshold)
        loss = pixel_loss + msssimloss * 0.2 + lpipsloss + dxyzsmoothloss + scaling_loss + dist_loss + normal_loss

        if not torch.isfinite(loss):
            print(f"\n[ITER {iteration}] WARNING: non-finite loss ({loss.item():.4f}), skipping optimizer step")
            for opt in gaussians.optimizers.values():
                opt.zero_grad(set_to_none=True)
            do_step = False
        else:
            loss.backward()

            # clip gradients to prevent explosion from high-lambda losses (lambda_dist=100)
            all_params = [p for opt in gaussians.optimizers.values()
                          for pg in opt.param_groups for p in pg['params']]
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            do_step = True

        # log part
        ema_vis_loss = 0.4 * pixel_loss.item() + 0.6 * ema_vis_loss
        ema_lpips_loss = 0.4 * lpipsloss.item() + 0.6 * ema_lpips_loss
        if iteration % 10 == 0:
            progress_bar.set_postfix({'pixel': f'{ema_vis_loss:.{4}f}', 'lpips': f'{ema_lpips_loss:.{4}f}'})
            progress_bar.update(10)
        if iteration == args.iterations:
            progress_bar.close()
        if iteration == args.iteration_sh_degree:
            gaussians.sh_degree += 1
            print(f'SH degree: {gaussians.sh_degree}')
        if iteration == args.texture_start:
            gaussians.start_texture_train()
            print('Texture training is started')
        if iteration == args.texture_stop:
            gaussians.stop_texture_train()
            print('Texture training is stopped')

        loss_dict = dict(pixel_loss=pixel_loss, lpips_loss=lpipsloss, dxyzsmooth_loss=dxyzsmoothloss,
                         scaling_loss=scaling_loss)
        training_report(scene, gaussians, iteration, args.test_iterations, loss_dict, background)

        # optimizer step
        if do_step:
            gaussians.optimizer_step()

        if iteration == args.iteration_dxyz_basis:
            gaussians.is_dxyz_bs = True
            print(f'[ITER {iteration}] Control point basis')

        if iteration == args.iteration_gsparam_basis:
            gaussians.is_gsparam_bs = True
            print(f'[ITER {iteration}] Gaussian property basis')

        # checkpoint
        if iteration in args.checkpoint_iterations:
            print("\n[ITER {}] Saving Checkpoint".format(iteration))
            save_data = gaussians.capture()
            save_data['iteration'] = iteration
            torch.save(save_data, path.join(args.out_dir, 'chkpnt' + str(iteration) + '.pth'))


report_cnt = 0
report_data = {}


@torch.no_grad()
def training_report(scene: Scene, gaussians: GaussianModel, iteration, test_iterations, loss_dict, background):
    global report_cnt, report_data
    tb_writer = scene.tb_writer
    report_cnt += 1
    for k, v in loss_dict.items():
        report_data[k] = report_data.get(k, 0) + v
    if report_cnt >= 10:
        for k, v in report_data.items():
            tb_writer.add_scalar(f'train_loss/{k}', v / report_cnt, iteration)
            report_data[k] = 0
        report_cnt = 0

    if iteration in test_iterations:
        torch.cuda.empty_cache()
        l1_test = 0.0
        psnr_test = 0.0

        rng = random.Random(0)
        write_idxs = [rng.randint(0, len(scene.trainset) - 1) for _ in range(10)]

        test_dataloader = DataLoader(
            dataset=scene.testset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

        cam_num = 0
        for cam in test_dataloader:
            cam = data_to_cam(cam, non_blocking=False)
            gaussians.smpl_poses = cam['pose']
            gaussians.Th, gaussians.Rh = cam['Th'], cam['Rh']

            image, alpha, info = gaussians.render(cam, background=background)
            image = torch.clamp(image, 0, 1)
            image_gt = cam['image']
            image_gt[~cam['mask']] = background

            l1_test += l1_loss(image, image_gt).mean().float()
            psnr_test += psnr(image, image_gt).mean().float()
            cam_num += 1

            if cam['idx'] in write_idxs:
                frame_id, cam_id = cam['frame_id'], cam['cam_id']
                tb_writer.add_images(f'train_view_{cam_id:02d}_{frame_id:06d}/render', image.permute(2, 0, 1)[None],
                                     global_step=iteration)
                if iteration == test_iterations[0]:
                    tb_writer.add_images(f'train_view_{cam_id:02d}_{frame_id:06d}/ground_truth',
                                         image_gt.permute(2, 0, 1)[None], global_step=iteration)

        psnr_test /= cam_num
        l1_test /= cam_num
        print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, 'train', l1_test, psnr_test))

        tb_writer.add_scalar('train/l1_loss', l1_test, iteration)
        tb_writer.add_scalar('train/psnr', psnr_test, iteration)
        tb_writer.add_scalar('total_gaussians', gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    cv2.setNumThreads(0)
    parser = ArgumentParser(description="Training script parameters")

    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=23458)
    pargs = parser.parse_args(sys.argv[1:])

    args = OmegaConf.load(pargs.config)
    args.data_dir, args.out_dir = pargs.data_dir, pargs.out_dir
    args.ip, args.port = pargs.ip, pargs.port
    os.makedirs(args.out_dir, exist_ok=True)

    OmegaConf.save(args, path.join(args.out_dir, 'config.yaml'))

    args.test_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)

    safe_state(False, args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(args)
