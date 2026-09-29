
import smplx
import numpy as np
from typing import Optional
import torch
import torch.nn.functional as F

class SMPLModel:
    model: smplx.SMPLX
    smpl_tpose = None
    smpl_bigpose = None
smpl = SMPLModel()

def init_smpl(pkl_path):
    smpl.model = smplx.SMPLX(
        model_path=pkl_path,
        use_pca=False,
        num_pca_comps=45, 
        flat_hand_mean=True, 
        batch_size=1)
    init_smpl_pose()

def init_smpl_pose():
    smpl.smpl_tpose = torch.zeros(165, dtype=torch.float32)
    smpl.smpl_bigpose = torch.zeros(165, dtype=torch.float32)
    smpl.smpl_bigpose[5] = np.deg2rad(25)
    smpl.smpl_bigpose[8] = np.deg2rad(-25) 


def interpolate_skinningfield(grid_info, points):
    grid, bbox_min, bbox_max = \
        grid_info['grid'], grid_info['bbox_min'], grid_info['bbox_max']
    grid, bbox_min, bbox_max = grid.cuda(), bbox_min.cuda(), bbox_max.cuda()
    points = points.cuda()

    points = (points - bbox_min) / (bbox_max - bbox_min) * 2 - 1

    points = points[:,[2,1,0]]
    intp_weights = F.grid_sample(
        input=grid.permute(3,0,1,2)[None],
        grid=points.reshape(1,1,1,-1,3),
        padding_mode='border',
        align_corners=True,
    )
    intp_weights = intp_weights.squeeze().permute(1,0)  # [P, 24]
    return intp_weights


import numba as nb
from scipy.spatial.transform import Rotation

def rigid_transform_tensor(pose, joint, parent):
    pose = pose.cpu().numpy()
    joint = joint.cpu().numpy()
    parent = parent.cpu().numpy()

    rots = Rotation.from_rotvec(pose.reshape(-1,3)).as_matrix()
    G = rigid_transform_numba(rots, joint, parent)

    G = torch.as_tensor(G).cuda(non_blocking=True)
    return G

@nb.njit(cache=True)
def rigid_transform_numba(R_pose, joint, parent):
    pose_num = len(parent)
    pa = parent
    joint_offset = np.copy(joint)
    G = np.zeros((pose_num, 4, 4), dtype=nb.float32)
    for i in range(len(G)):
        for j in range(4):
            G[i,j,j] = 1.0

    joint_offset[1:] = joint[1:] - joint[pa[1:]]

    G[:,:3,:3] = R_pose
    G[:,:3,3] = joint_offset

    for i in range(1, pose_num):
        G[i] = G[pa[i]] @ G[i]
    return G
