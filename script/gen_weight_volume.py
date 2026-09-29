# code from AnimatableGaussians https://github.com/lizhe00/AnimatableGaussians/tree/master/gen_data

import os
import glob
import numpy as np
import tqdm
import math

import smplx
import torch
import trimesh
from os import path

tmp_dir = './tmp'
os.makedirs(tmp_dir, exist_ok = True)
depth = 7


def compute_lbs_grad(cano_smpl: trimesh.Trimesh, vertex_lbs: np.ndarray):
    vertices = cano_smpl.vertices.astype(np.float32)
    normals = cano_smpl.vertex_normals.copy().astype(np.float32)
    faces = cano_smpl.faces.astype(np.int64)

    normals /= np.linalg.norm(normals, axis = 1, keepdims = True)
    tx = np.cross(normals, np.array([[0, 0, 1]], np.float32))
    tx /= np.linalg.norm(tx, axis = 1, keepdims = True)
    ty = np.cross(normals, tx)

    vnum = vertices.shape[0]
    fnum = faces.shape[0]
    jnum = vertex_lbs.shape[1]
    lbs_grad_tx = np.zeros((vnum, jnum), np.float32)
    lbs_grad_ty = np.zeros((vnum, jnum), np.float32)
    for vidx in tqdm.tqdm(range(vnum)):
        v = vertices[vidx]
        for jidx in range(jnum):
            for neighbor_vidx in cano_smpl.vertex_neighbors[vidx]:
                vn = vertices[neighbor_vidx]
                pos_diff = vn - v
                pos_diff_norm = np.linalg.norm(pos_diff)
                val_diff = vertex_lbs[neighbor_vidx, jidx] - vertex_lbs[vidx, jidx]
                val_diff /= pos_diff_norm
                pos_diff /= pos_diff_norm

                lbs_grad_tx[vidx, jidx] += val_diff * np.dot(pos_diff, tx[vidx])
                lbs_grad_ty[vidx, jidx] += val_diff * np.dot(pos_diff, ty[vidx])

            lbs_grad_tx[vidx, jidx] /= float(len(cano_smpl.vertex_neighbors[vidx]))
            lbs_grad_ty[vidx, jidx] /= float(len(cano_smpl.vertex_neighbors[vidx]))

    lbs_grad = lbs_grad_tx[:, :, None] * tx[:, None] + lbs_grad_ty[:, :, None] * ty[:, None]  # (V, J, 3)
    for jid in tqdm.tqdm(range(jnum)):
        out_fn_grad = os.path.join(tmp_dir, f"cano_data_lbs_grad_{jid:02d}.xyz")
        out_fn_val = os.path.join(tmp_dir, f"cano_data_lbs_val_{jid:02d}.xyz")

        out_data_grad = np.concatenate([vertices, lbs_grad[:, jid]], 1)
        out_data_val = np.concatenate([vertices, vertex_lbs[:, jid:jid+1]], 1)
        np.savetxt(out_fn_grad, out_data_grad, fmt="%.8f")
        np.savetxt(out_fn_val, out_data_val, fmt="%.8f")


def solve(num_joints, point_interpolant_exe):
    for jid in range(num_joints):
        print('Solving joint %d' % jid)
        cmd = f'{point_interpolant_exe} ' + \
            f'--inValues {os.path.join(tmp_dir, f"cano_data_lbs_val_{jid:02d}.xyz")} ' + \
            f'--inGradients {os.path.join(tmp_dir, f"cano_data_lbs_grad_{jid:02d}.xyz")} ' + \
            f'--gradientWeight 0.05 --dim 3 --verbose ' + \
            f'--grid {os.path.join(tmp_dir, f"grid_{jid:02d}.grd")} ' + \
            f'--depth {depth} '

        os.system(cmd)

class Config:
    # SMPL related
    cano_smpl_pose = np.zeros(75, dtype = np.float32)
    cano_smpl_pose[3+3*1+2] = math.radians(25)
    cano_smpl_pose[3+3*2+2] = math.radians(-25)
    cano_smpl_pose = torch.from_numpy(cano_smpl_pose)
    cano_smpl_transl = cano_smpl_pose[:3]
    cano_smpl_global_orient = cano_smpl_pose[3:6]
    cano_smpl_body_pose = cano_smpl_pose[6:69]

config = Config()

@torch.no_grad()
def calc_cano_weight_volume(data_dir, pkl_path, gender = 'neutral'):
    if 'SMPLX' in pkl_path:
        smpl_params = np.load(path.join(data_dir, 'smpl_params.npz'), allow_pickle=True)
        smpl_shape = torch.as_tensor(smpl_params['betas'][0])
        smpl_model = smplx.SMPLX(model_path=pkl_path, use_pca=False, num_pca_comps=45, flat_hand_mean=True, batch_size=1)
    else:
        raise NotImplementedError

    # def get_grid_points(bounds, res):
    #     # voxel_size = (bounds[1] - bounds[0]) / (np.array(res, np.float32) - 1)
    #     # x = np.arange(bounds[0, 0], bounds[1, 0] + voxel_size[0], voxel_size[0])
    #     # y = np.arange(bounds[0, 1], bounds[1, 1] + voxel_size[1], voxel_size[1])
    #     # z = np.arange(bounds[0, 2], bounds[1, 2] + voxel_size[2], voxel_size[2])
    #     x = np.linspace(bounds[0, 0], bounds[1, 0], res[0])
    #     y = np.linspace(bounds[0, 1], bounds[1, 1], res[1])
    #     z = np.linspace(bounds[0, 2], bounds[1, 2], res[2])

    #     pts = np.stack(np.meshgrid(x, y, z, indexing = 'ij'), axis = -1)
    #     return pts

    if isinstance(smpl_model, smplx.SMPLX):
        cano_smpl = smpl_model.forward(betas = smpl_shape[None],
                                       global_orient = config.cano_smpl_global_orient[None],
                                       transl = config.cano_smpl_transl[None],
                                       body_pose = config.cano_smpl_body_pose[None])
    elif isinstance(smpl_model, smplx.SMPL):
        cano_smpl = smpl_model.forward(betas = smpl_shape[None],
                                       global_orient = config.cano_smpl_global_orient[None],
                                       transl = config.cano_smpl_transl[None],
                                       body_pose = config.cano_smpl_pose[6:][None])
    else:
        raise TypeError('Not support this SMPL type.')

    cano_smpl_trimesh = trimesh.Trimesh(
        cano_smpl.vertices[0].cpu().numpy(),
        smpl_model.faces,
        process = False
    )

    compute_lbs_grad(cano_smpl_trimesh, smpl_model.lbs_weights.cpu().numpy())
    solve(smpl_model.lbs_weights.shape[-1], "../bins/PointInterpolant")

    ### NOTE concatenate all grids
    fn_list = sorted(list(glob.glob(os.path.join(tmp_dir, 'grid_*.grd'))))

    grids = []
    import array
    for fn in fn_list:
        with open(fn, 'rb') as f:
            bytes = f.read()
        grid_res = 2 ** depth
        grid_header_len = len(bytes) - grid_res ** 3 * 8
        grid_np = np.array(array.array('d', bytes[grid_header_len:])).reshape(grid_res, grid_res, grid_res)
        grids.append(grid_np)

    grids_all = np.stack(grids, 0)
    grids_all = np.clip(grids_all, 0.0, 1.0)
    grids_all = grids_all / grids_all.sum(0)[None]
    # print(grids_all.shape)
    # np.save(join(data_templates_path, subject, subject + '_cano_lbs_weights_grid_float32.npy'), grids_all.astype(np.float32))
    diff_weights = grids_all.transpose((3, 2, 1, 0))  # convert to xyz
    min_xyz = cano_smpl_trimesh.vertices.min(0).astype(np.float32)
    max_xyz = cano_smpl_trimesh.vertices.max(0).astype(np.float32)
    max_len = 1.1 * (max_xyz - min_xyz).max()
    center = 0.5 * (min_xyz + max_xyz)
    volume_bounds = np.stack(
        [center - 0.5 * max_len, center + 0.5 * max_len], 0
    )

    min_xyz[:2] -= 0.05
    max_xyz[:2] += 0.05
    min_xyz[2] -= 0.15
    max_xyz[2] += 0.15
    smpl_bounds = np.stack(
        [min_xyz, max_xyz], 0
    )
    res = diff_weights.shape[:3]

    os.makedirs(path.join(data_dir, 'gaussian'), exist_ok=True)

    data = dict(
        grid = diff_weights.astype(np.float32),
        bbox_min = volume_bounds[0],
        bbox_max = volume_bounds[1],
        grid_dims = np.array(res),
        grid_resolution = (volume_bounds[1,0] - volume_bounds[0,0]) / (np.array(res)[0] - 1),
    )
    # for key in data:
    #     if key in ['grid_resolution']: continue
    #     data[key] = torch.as_tensor(data[key])
    #     print(key, data[key].dtype, data[key].shape)
    np.savez(path.join(data_dir, 'gaussian/lbs_weights_grid.npz'), **data)


if __name__ == '__main__':
    from argparse import ArgumentParser
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--data_dir', type = str, help = 'Dataset path.')
    arg_parser.add_argument('--smpl_path', type = str, help = 'Smpl path.')
    args = arg_parser.parse_args()

    data_dir = path.expanduser(args.data_dir)
    smpl_path = path.expanduser(args.smpl_path)

    calc_cano_weight_volume(data_dir, smpl_path, gender = 'neutral')
