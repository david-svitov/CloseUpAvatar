
import os
from os import path
import torch
import numpy as np
import json
from torch.utils.data import DataLoader
import open3d as o3d

from tensorboardX import SummaryWriter

from utils.config_utils import Config
from utils.smpl_utils import init_smpl, smpl
from utils.general_utils import serialize_to_list, storePly, fetchPly
from utils.graphics_utils import rand_point_on_mesh
from scene.gaussian_model import GaussianModel
from scene.dataset import get_dataset_type, AVRexDataset

class Scene:
    gaussians: GaussianModel
    tb_writer: SummaryWriter
    trainset: AVRexDataset
    testset: AVRexDataset
    trainloader: DataLoader
    scene_scale = None
    def __init__(self, args: Config, gaussians : GaussianModel):
        tb_writer = SummaryWriter(args.out_dir)
        init_smpl(args.smpl_pkl_path)

        # dataset
        frame_ids = np.arange(args.begin_ith_frame, args.begin_ith_frame+args.frame_interval*args.num_train_frame, args.frame_interval).tolist()
        cam_ids = np.array(args.train_cam_ids).tolist()
        image_scaling = args.image_scaling
        DatasetType = get_dataset_type(args.data_dir)
        print(f'Dataset: {DatasetType.__name__}')
        trainset = DatasetType(
            datadir=args.data_dir,
            frame_ids=frame_ids,
            cam_ids=cam_ids,
            background=np.array(args.background),
            image_scaling=image_scaling,
            is_in_memory=args.data_in_memory,
        )
        test_frame_ids = np.arange(args.test.begin_ith_frame, args.test.begin_ith_frame+args.test.frame_interval*args.test.num_frame, args.test.frame_interval).tolist()
        test_cam_ids = np.array(args.test.cam_ids).tolist()
        testset = DatasetType(
            datadir=args.data_dir,
            frame_ids=test_frame_ids,
            cam_ids=test_cam_ids,
            background=np.array(args.background),
            image_scaling=image_scaling,
        )
        print(f'Training images: {len(trainset)} Test images: {len(testset)}')

        # dataloader
        trainloader = DataLoader(
            dataset=trainset,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )

        # collect all the poses and cams from dataset, and dump them to json file
        all_poses, all_Th, all_Rh = {}, {}, {}
        cam_list, pose_list = {}, {}
        trainset.is_load_image = False
        beta = trainset[0]['beta']

        for data in trainset:
            cam_id, frame_id = data['cam_id'], data['frame_id']
            if str(frame_id) not in all_poses:
                all_poses[str(frame_id)] = data['pose']
                all_Th[str(frame_id)] = data['Th']
                all_Rh[str(frame_id)] = data['Rh']
            
                pose_list[str(frame_id)] = dict(frame_id=frame_id, pose=data['pose'], Th=data['Th'], Rh=data['Rh'])

            if str(cam_id) not in cam_list:
                cam_list[str(cam_id)] = dict(cam_id=cam_id, w2c=data['w2c'], K=data['K'])
        trainset.is_load_image = True

        cam_list = sorted(cam_list.values(), key=lambda x: x['cam_id'])
        pose_list = sorted(pose_list.values(), key=lambda x: x['frame_id'])
        with open(path.join(args.out_dir, "poses.json"), 'w') as file:
            json.dump(serialize_to_list(pose_list), file)    
        with open(path.join(args.out_dir, "cameras.json"), 'w') as file:
            json.dump(serialize_to_list(cam_list), file)
        
        # skinning weights
        os.makedirs(path.join(args.data_dir, 'gaussian'), exist_ok=True)
        weights_grid_path = path.join(args.data_dir, 'gaussian/lbs_weights_grid.npz')
        if not path.exists(weights_grid_path):
            raise FileExistsError
        grid_info = dict(np.load(weights_grid_path, allow_pickle=True))

        # initialize gaussian model
        scene_scale = trainset.get_scene_scale(args.data_dir) * 1.1
        tpose_model = smpl.model(betas=beta[None], body_pose=smpl.smpl_tpose[None,3*1:22*3])
        t_joints = tpose_model.joints.detach().numpy()[0,:smpl.model.NUM_JOINTS+1]
        xyz_path = path.join(args.data_dir, 'gaussian/init_body_points.ply')

        temp_path = path.join(args.data_dir, 'gaussian/template.ply')
        if not path.exists(temp_path):
            print('No template found, using SMPLX mesh')
            bigpose_model = smpl.model(betas=beta[None], body_pose=smpl.smpl_bigpose[None,3*1:22*3])
            verts = bigpose_model.vertices[0].detach().float().numpy()
            faces = smpl.model.faces

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            o3d.io.write_triangle_mesh(temp_path, mesh)

        mesh = o3d.io.read_triangle_mesh(path.join(args.data_dir, 'gaussian/template.ply'))
        verts = np.array(mesh.vertices).astype(np.float32)
        faces = np.array(mesh.triangles).astype(np.float32)

        if path.exists(xyz_path):
            xyz = np.array(fetchPly(xyz_path)[0], dtype=np.float32)
        else:
            print('Initialize Gaussians on Mesh...')
            xyz = rand_point_on_mesh(verts, faces, pts_num=args.init_num_gs)
            storePly(xyz_path, xyz, np.zeros_like(xyz))

        xyz_ft = rand_point_on_mesh(verts, faces, pts_num=args.num_features, init_factor=7)
        xyz_vt = rand_point_on_mesh(verts, faces, pts_num=args.num_verts, init_factor=7)

        gaussians.create_from_pcd(
            xyz=xyz,
            t_joints=t_joints,
            joint_parents=smpl.model.parents,
            lbs_weights_grid_info=grid_info, 
            all_poses=all_poses,
            xyz_ft=xyz_ft,
            xyz_vt=xyz_vt,
        )

        self.tb_writer = tb_writer
        self.gaussians = gaussians
        self.scene_scale = scene_scale
        self.trainset = trainset
        self.testset = testset
        self.trainloader = trainloader
        self.trainloader_len = len(trainset)