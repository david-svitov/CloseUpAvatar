
import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F
from diff_bbsplat_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from pytorch3d.ops import knn_points
from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply
from scipy.spatial.transform import Rotation
from torch import nn
from torch.func import stack_module_state
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ExponentialLR
from gsplat import spherical_harmonics

from scene.mlp import MLP, vmap_mlp
from utils.config_utils import Config
from utils.graphics_utils import focal2fov, getProjectionMatrix, depth_to_normal
from utils.sh_utils import RGB2SH
from utils.smpl_utils import smpl, interpolate_skinningfield, rigid_transform_tensor, rigid_transform_numba


class GaussianModel:

    def setup_functions(self):
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.additive_opacity_activation = torch.tanh
        self.inverse_additive_opacity_activation = torch.atanh
        self.inverse_opacity_activation = torch.logit

        self.rotation_activation = F.normalize

        self.color_activation = torch.sigmoid
        self.additive_color_activation = torch.tanh
        self.inverse_additive_color_activation = torch.atanh
        self.inverse_color_activation = torch.logit

    def __init__(self):

        self._xyz = torch.empty(0)
        self.xyz_offset = torch.empty(0)
        self.dxyz_vt = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._sh0 = torch.empty(0)
        self._shN = torch.empty(0)
        self.sh_degree = 0
        self._texture_color = torch.empty(0)
        self._texture_alpha = torch.empty(0)
        self._texture_color_precalc = None
        self._texture_alpha_precalc = None

        self.xyz_vt = torch.empty(0)
        self.xyz_ft = torch.empty(0)

        # Load predefined texture shape
        self._texture_alpha_init = cv2.imread("assets/alpha_init_gaussian_small.png")[..., 0] / 255.0
        self.stored_rescaled_init = None
        self._texture_scales = [16, 32] #8, 16, 32

        # basis property definition
        self.num_vt_basis = 15     # Control point basis number
        self.num_basis = 15        # Gaussian property basis number

        self.encoder_feat_params = None
        self.encoder_feat_model_meta = None

        self.dxyz_bs = torch.empty(0)
        self.sh0_bs = torch.empty(0)
        self.shN_bs = torch.empty(0)
        self.scaling_bs = torch.empty(0)
        self.rotation_bs = torch.empty(0)
        self.opacity_bs = torch.empty(0)

        # lbs weights
        self._weights = None

        # pose
        self._Rh = torch.empty(0)
        self._Th = torch.empty(0)
        self.Ac_inv = torch.empty(0)
        self._smpl_poses = torch.empty(0)
        self.smpl_poses_cuda = torch.empty(0)
        self.t_joints = torch.empty(0)
        self.joint_parents = torch.empty(0)

        self.all_poses = torch.empty(0)

        # cache
        self.cache_dict = {}

        # optimizer
        self.optimizers = None
        self.schedulers = None

        # knn
        self.nbr_gs = torch.empty(0)
        self.nbr_gs_invdist = torch.empty(0)
        self.nbr_vt = torch.empty(0)
        self.nbr_gsft = torch.empty(0)
        self.nbr_vtft = torch.empty(0)
        self.nbr_gsft_wght = torch.empty(0)
        self.nbr_vtft_wght = torch.empty(0)

        # misc
        self.scene_scale = None
        self.is_dxyz_bs = False     # whether to use control point basis
        self.is_gsparam_bs = False  # whether to use Gaussian property basis

        self.is_test = False        # whether to use PCA 

        self.setup_functions()

    def capture(self):
        data = {
            '_xyz': self._xyz,
            'xyz_offset': self.xyz_offset,
            'dxyz_vt': self.dxyz_vt,
            '_scaling': self._scaling,
            '_rotation': self._rotation,
            '_sh0': self._sh0,
            '_shN': self._shN,
            'sh_degree': self.sh_degree,
            '_texture_color': self._texture_color,
            '_texture_alpha': self._texture_alpha,

            '_weights': self.get_weights,

            't_joints': self.t_joints,
            'all_poses': self.all_poses,
            'joint_parents': self.joint_parents,

            'nbr_gs_invdist': self.nbr_gs_invdist,
            'nbr_gs': self.nbr_gs,
            'nbr_vt': self.nbr_vt,
            'nbr_gsft': self.nbr_gsft,
            'nbr_vtft': self.nbr_vtft,
            'nbr_gsft_wght': self.nbr_gsft_wght,
            'nbr_vtft_wght': self.nbr_vtft_wght,

            'xyz_vt': self.xyz_vt,
            'xyz_ft': self.xyz_ft,

            'num_vt_basis': self.num_vt_basis,
            'num_basis': self.num_basis,

            'encoder_feat_params': self.encoder_feat_params,
            'encoder_feat_model_meta': self.encoder_feat_model_meta,

            'dxyz_bs': self.dxyz_bs,
            'sh0_bs': self.sh0_bs,
            'shN_bs': self.shN_bs,
            'scaling_bs': self.scaling_bs,
            'rotation_bs': self.rotation_bs,
            'opacity_bs': self.opacity_bs,

            'is_dxyz_bs': self.is_dxyz_bs,
            'is_gsparam_bs': self.is_gsparam_bs,
        }
        return data
    
    def restore(self, data):
        def loader(s):
            if s in data: return data[s]
            else: print(f'NO DATA {s}!')
            return None

        self._xyz = data['_xyz']
        self.xyz_offset = data['xyz_offset']
        self.dxyz_vt = data['dxyz_vt']
        self._rotation = data['_rotation']
        self._scaling = data['_scaling']
        self._sh0 = data['_sh0']
        self._shN = loader('_shN')
        self.sh_degree = data['sh_degree']
        self._texture_color = data['_texture_color']
        self._texture_alpha = data['_texture_alpha']

        self._texture_color_precalc = self.get_texture_color
        self._texture_alpha_precalc = self.get_texture_alpha

        self._weights = data['_weights']

        self.t_joints = loader('t_joints')
        self.all_poses = loader('all_poses')
        self.joint_parents = loader('joint_parents')

        self.nbr_gs = loader('nbr_gs')
        self.nbr_vt = loader('nbr_vt')
        self.nbr_gs_invdist = loader('nbr_gs_invdist')
        self.nbr_gsft = loader('nbr_gsft')
        self.nbr_vtft = loader('nbr_vtft')
        self.nbr_gsft_wght = loader('nbr_gsft_wght')
        self.nbr_vtft_wght = loader('nbr_vtft_wght')

        self.xyz_vt = loader('xyz_vt')
        self.xyz_ft = loader('xyz_ft')

        self.num_vt_basis = loader('num_vt_basis')
        self.num_basis = loader('num_basis')

        self.encoder_feat_params = loader('encoder_feat_params')
        self.encoder_feat_model_meta = loader('encoder_feat_model_meta')

        self.dxyz_bs = loader('dxyz_bs')
        self.sh0_bs = loader('sh0_bs')
        self.shN_bs = loader('shN_bs')
        self.scaling_bs = loader('scaling_bs')
        self.rotation_bs = loader('rotation_bs') 
        self.opacity_bs = loader('opacity_bs')

        self.is_dxyz_bs = loader('is_dxyz_bs')
        self.is_gsparam_bs = loader('is_gsparam_bs')

        self.init()

    def init(self):
        self.init_body() 
        self.reset_pose()   

    @property
    def get_cano_scaling(self):
        if 'get_cano_scaling' in self.cache_dict: return self.cache_dict['get_cano_scaling'] 
        if not self.is_gsparam_bs: 
            scaling = self.scaling_activation(self._scaling)
        else:
            features = self.get_encoded_feature_gsparam_weight
            dscaling = torch.einsum('nc,ncl->nl', features, self.scaling_bs)

            scaling = self._scaling + dscaling
            scaling = self.scaling_activation(scaling)
        
        self.cache_dict['get_cano_scaling'] = scaling
        return scaling
    
    @property
    def get_weights(self):
        if self._weights is None:
            xyz = self._xyz
            weights = interpolate_skinningfield(self.weights_grid_info, xyz)
            self._weights = weights
        else:
            weights = self._weights
        return weights

    @property
    def get_rigid_transform(self):
        if 'get_rigid_transform' in self.cache_dict: return self.cache_dict['get_rigid_transform']
        pose = self.smpl_poses.numpy()
        joints = self.t_joints.numpy()
        parent = self.joint_parents.numpy()
        Ac_inv = self.Ac_inv.numpy()

        rots = Rotation.from_rotvec(pose.reshape(-1,3)).as_matrix().astype(np.float32)
        A = rigid_transform_numba(rots, joints, parent)
        G = np.matmul(A, Ac_inv)

        data = [torch.as_tensor(d).cuda(non_blocking=True) for d in [rots, G]]
        self.cache_dict['get_rigid_transform'] = data
        return data

    @property
    def get_Gweights(self):
        if 'get_Gweights' in self.cache_dict: return self.cache_dict['get_Gweights']
        
        G = self.get_rigid_transform[1]
        G_weight = torch.einsum('vp,pij->vij', self.get_weights, G)

        self.cache_dict['get_Gweights'] = G_weight
        return G_weight

    @property
    def get_cano_rotation(self):
        if not self.is_gsparam_bs: 
            rotation = self.rotation_activation(self._rotation)
        else:
            features = self.get_encoded_feature_gsparam_weight
            drotation = torch.einsum('nc,ncl->nl', features, self.rotation_bs)

            rotation = self._rotation + drotation
            rotation = self.rotation_activation(rotation)

        return rotation

    @property
    def get_joint_features(self):

        if self.is_test:
            sigma_pca = 2.0
            features = self.smpl_poses_cuda[1*3:22*3][None]
            lowdim_pose_conds = self.pca.transform(features)
            std = self.pca_std
            lowdim_pose_conds = torch.maximum(lowdim_pose_conds, -sigma_pca * std)
            lowdim_pose_conds = torch.minimum(lowdim_pose_conds, sigma_pca * std)
            features = self.pca.inverse_transform(lowdim_pose_conds).reshape(-1)
        else:
            features = self.smpl_poses_cuda[3:3*22]

        return features

    @torch.no_grad()
    def prepare_test(self):
        pose_set = []
        for k, v in self.all_poses.items():
            pose_set.append(v[1*3:22*3].detach())     
        N_pose = len(pose_set)
        pose_set = torch.stack(pose_set, dim=0).reshape(N_pose,21,3).cpu().numpy()
        features = pose_set.reshape(N_pose, -1)

        pca_num = 20

        features = torch.as_tensor(features).cuda()
        from torch_pca import PCA
        self.pca = PCA(n_components=pca_num)
        self.pca.fit(features)
        self.pca_std = torch.sqrt(self.pca.explained_variance_)

        print(f'Use PCA components: {pca_num}')

    @property
    def get_encoded_feature(self):
        if 'get_encoded_feature' in self.cache_dict: return self.cache_dict['get_encoded_feature']
        features = self.get_joint_features
        N_feat = len(self.encoder_feat_params['layers.0.weight'])
        features = features.tile([N_feat, 1])
        features = vmap_mlp(self.encoder_feat_params, features)

        self.cache_dict['get_encoded_feature'] = features
        return features

    @property
    def get_encoded_feature_gsparam_weight(self):
        if 'get_encoded_feature_gsparam_weight' in self.cache_dict: return self.cache_dict['get_encoded_feature_gsparam_weight']
        features = self.get_encoded_feature[...,:self.num_basis]
        features = torch.einsum('nrc,nr->nc', features[self.nbr_gsft], self.nbr_gsft_wght)

        self.cache_dict['get_encoded_feature_gsparam_weight'] = features
        return features

    @property
    def get_dxyz_vt(self):
        if 'get_dxyz_vt' in self.cache_dict: return self.cache_dict['get_dxyz_vt']
        if not self.is_dxyz_bs: return self.dxyz_vt

        features = self.get_encoded_feature[...,self.num_basis:]
        features = torch.einsum('nrc,nr->nc', features[self.nbr_vtft], self.nbr_vtft_wght)

        dxyz_vt = torch.einsum('vc,vcl->vl', features, self.dxyz_bs)

        dxyz_vt = self.dxyz_vt + dxyz_vt
        self.cache_dict['get_dxyz_vt'] = dxyz_vt

        return dxyz_vt

    @property
    def get_dxyz(self):
        if 'get_dxyz' in self.cache_dict: return self.cache_dict['get_dxyz']

        dxyz = torch.sum(self.nbr_gs_invdist[...,None] * self.get_dxyz_vt[self.nbr_gs], dim=1) / torch.sum(self.nbr_gs_invdist, dim=-1)[...,None]
        self.cache_dict['get_dxyz'] = dxyz
        return dxyz
    
    @property
    def get_cano_xyz(self):
        if 'get_cano_xyz' in self.cache_dict: return self.cache_dict['get_cano_xyz']
        xyz = self._xyz + self.get_dxyz + torch.tanh(self.xyz_offset) * 0.008   # A trick to allow Gaussians to move freely within a small range
        self.cache_dict['get_cano_xyz'] = xyz
        return xyz

    @property
    def get_xyz(self):
        if 'get_xyz' in self.cache_dict: return self.cache_dict['get_xyz']
        xyz = self.get_cano_xyz
        xyz = torch.einsum('vij,vj->vi', self.get_Gweights, F.pad(xyz,(0,1),value=1))[:,:3]
        if self.Rh is not None: xyz = torch.einsum('ij,vj->vi', self.Rh, xyz) 
        xyz = xyz + self.Th

        self.cache_dict['get_xyz'] = xyz
        return xyz

    @property
    def get_sh(self):
        if 'get_sh' in self.cache_dict: return self.cache_dict['get_sh']

        if self.sh_degree == 0: 
            sh = self._sh0
        else:
            sh = torch.cat([self._sh0, self._shN], dim=1)

        if self.is_gsparam_bs:

            features = self.get_encoded_feature_gsparam_weight
            dsh0 = torch.einsum('nc,ncxy->nxy', features, self.sh0_bs)
            if self.sh_degree == 0: 
                dsh = dsh0
            else: 
                dshN = torch.einsum('nc,ncxy->nxy', features, self.shN_bs)
                dsh = torch.cat([dsh0, dshN], dim=1)

            sh = sh + dsh

        self.cache_dict['get_sh'] = sh
        return sh

    def get_color(self, cam_pos):
        if 'get_color' in self.cache_dict: return self.cache_dict['get_color']

        if self.sh_degree > 0:
            rots = self.get_Gweights[:,:3,:3]
            # with torch.set_grad_enabled(False):
            #     rots = polar_decomposition_newton_schulz(rots)

            dirs = F.normalize(cam_pos - self.get_xyz, dim=-1)
            invrots = rots.transpose(-1,-2)
            dirs = torch.einsum('nij,nj->ni',invrots, dirs)
        else:
            dirs = torch.ones_like(self._xyz)

        sh = self.get_sh
        color = spherical_harmonics(self.sh_degree, dirs, sh)
        color = torch.clamp_min(color + 0.5, 0)

        self.cache_dict['get_color'] = color

        return color

    @property
    def get_texture_color(self):
        if self._texture_color_precalc is not None:
            return self._texture_color_precalc

        activated_colors = [self.color_activation(self._texture_color[0])]
        for tensor in self._texture_color[1:]:
            tensor = self.additive_color_activation(tensor)
            activated_colors.append(tensor)
        return activated_colors

    @property
    def get_texture_alpha(self):
        if self._texture_alpha_precalc is not None:
            return self._texture_alpha_precalc

        activated_alphas = [self.opacity_activation(self._texture_alpha[0])]
        for tensor in self._texture_alpha[1:]:
            tensor = self.additive_opacity_activation(tensor)
            activated_alphas.append(tensor)
        return activated_alphas

    def start_texture_train(self):
        for group in self.optimizers["texture_alpha"].param_groups:
            group["lr"] = self._texture_alpha_lr
        for group in self.optimizers["texture_color"].param_groups:
            group["lr"] = self._texture_color_lr

    def stop_texture_train(self):
        for group in self.optimizers["texture_alpha"].param_groups:
            group["lr"] = 0
        for group in self.optimizers["texture_color"].param_groups:
            group["lr"] = 0

    def compute_vertex_normals(self, vertices, neighbor_indices):
        N = vertices.shape[0]
        K = neighbor_indices.shape[1]

        neighbors = vertices[neighbor_indices.view(-1)].view(N, K, 3)
        vectors = neighbors - vertices.unsqueeze(1)
        normals = torch.zeros_like(vertices)
        for i in range(K - 1):
            edge1 = vectors[:, i]
            edge2 = vectors[:, (i + 1) % K]
            normals += torch.cross(edge1, edge2)

        normals = F.normalize(normals, dim=1)
        return normals

    def compute_rotation_quaternions(self, vertices, neighbor_indices, up_direction=torch.tensor([0., 1., 0.])):
        device = vertices.device
        up_direction = up_direction.to(device)

        normals = self.compute_vertex_normals(vertices, neighbor_indices)
        tangent = torch.cross(up_direction.expand_as(normals), normals)
        tangent = F.normalize(tangent, dim=1)

        binormal = torch.cross(normals, tangent)
        binormal = F.normalize(binormal, dim=1)

        rotation_matrix = torch.stack([tangent, binormal, normals], dim=2)
        quaternions = matrix_to_quaternion(rotation_matrix)

        return quaternions


    def create_from_pcd(self, xyz=None, t_joints=None, joint_parents=None, all_poses=None, lbs_weights_grid_info=None, xyz_vt=None, xyz_ft=None):
        xyz = torch.as_tensor(xyz).float().cuda() # [N,3]
        N = xyz.shape[0]
        print("Number of points at initialization : ", N)

        init_opacity = 0.8
        init_color = 0.5

        # Initialize the GS size to be the average dist of the 3 nearest neighbors
        dist2, nn_indeces, _ = knn_points(xyz[None], xyz[None], K=4)
        dist2_avg = dist2[0, :, 1:].mean(dim=-1, keepdim=True)
        scale = self.scaling_inverse_activation(torch.sqrt(dist2_avg)).tile([1,2])  # [N,3]

        nn_indeces = nn_indeces[0, :, 1:].contiguous()
        rotation = self.compute_rotation_quaternions(xyz, nn_indeces)

        sh0 = torch.full((N, 1, 3), RGB2SH(init_color)).float().cuda() 
        shN = torch.zeros((N, 3, 3)).float().cuda()
        xyz_offset = torch.zeros_like(xyz)

        self._xyz = xyz
        self.xyz_offset = nn.Parameter(xyz_offset.requires_grad_(True))
        self._rotation = nn.Parameter(rotation.requires_grad_(True))
        self._scaling = nn.Parameter(scale.requires_grad_(True))
        self._sh0 = nn.Parameter(sh0.requires_grad_(True))
        self._shN = nn.Parameter(shN.requires_grad_(True))

        self._texture_alpha = nn.ParameterList([])
        self._texture_color = nn.ParameterList([])
        for i, size in enumerate(self._texture_scales):
            # Setup trainable alpha textures
            if i == 0:
                alpha_init = cv2.resize(self._texture_alpha_init, (size, size))
                rescaled_init = torch.tensor([alpha_init], dtype=torch.float, device="cuda") + 1e-6
                self.stored_rescaled_init = rescaled_init.clone()
                rescaled_init = self.inverse_opacity_activation(rescaled_init)
            else:
                rescaled_init = torch.zeros([1, size, size], dtype=torch.float, device="cuda")
            texture_alpha = rescaled_init.repeat(N, 1, 1)
            texture_alpha = nn.Parameter(texture_alpha.requires_grad_(True))
            self._texture_alpha.append(texture_alpha)

            # Setup trainable RGB textures
            texture_size = texture_alpha.shape[-1]
            if i == 0:
                texture_color = torch.ones([N, 3, texture_size, texture_size]) * 0.1
                texture_color = self.inverse_color_activation(texture_color)
            else:
                texture_color = torch.zeros([N, 3, texture_size, texture_size])
            texture_color = texture_color.to("cuda").type(torch.float)
            texture_color = nn.Parameter(texture_color.requires_grad_(True))
            self._texture_color.append(texture_color)

        self.t_joints = torch.as_tensor(t_joints).detach().float().cpu()
        self.joint_parents = torch.as_tensor(joint_parents).detach().cpu()

        for key in all_poses: all_poses[key] = torch.as_tensor(all_poses[key]).float().cpu()
        self.all_poses = all_poses

        ginfo = lbs_weights_grid_info
        for key in ['grid', 'bbox_min', 'bbox_max', 'grid_dims']: ginfo[key] = torch.as_tensor(ginfo[key]).detach().cuda()
        self.weights_grid_info = ginfo

        # Pose encoder
        models = [MLP(layers_size_list=[63, 512, 256, 256, 256, self.num_basis+self.num_vt_basis]) for i in range(len(xyz_ft))]
        params, _ = stack_module_state(models)
        self.encoder_feat_model_meta = MLP(layers_size_list=[63, 512, 256, 256, 256, self.num_basis+self.num_vt_basis]).to('meta')
        for k, v in params.items():
            params[k] = nn.Parameter(v.cuda().requires_grad_(True))
        self.encoder_feat_params = params

        # basis
        dxyz_bs = torch.zeros((len(xyz_vt), self.num_vt_basis, 3)).float().cuda()
        sh0_bs = torch.zeros((N, self.num_basis, 1, 3)).float().cuda()
        shN_bs = torch.zeros((N, self.num_basis, 3, 3)).float().cuda()
        scaling_bs = torch.zeros((N, self.num_basis, 2)).float().cuda()
        rotation_bs = torch.zeros((N, self.num_basis, 4)).float().cuda()
        opacity_bs = torch.zeros((N, self.num_basis)).float().cuda()
        for data in [dxyz_bs, sh0_bs, scaling_bs, rotation_bs, opacity_bs]:
            nn.init.uniform_(data[0], -0.002, 0.002)
            data[1:] = data[0]
        self.dxyz_bs = nn.Parameter(dxyz_bs.requires_grad_(True))
        self.sh0_bs = nn.Parameter(sh0_bs.requires_grad_(True))
        self.shN_bs = nn.Parameter(shN_bs.requires_grad_(True))
        self.scaling_bs = nn.Parameter(scaling_bs.requires_grad_(True))
        self.rotation_bs = nn.Parameter(rotation_bs.requires_grad_(True))
        self.opacity_bs = nn.Parameter(opacity_bs.requires_grad_(True))

        xyz_ft = torch.as_tensor(xyz_ft).float().cuda()
        xyz_vt = torch.as_tensor(xyz_vt).float().cuda()
        self.dxyz_vt = nn.Parameter(torch.zeros_like(xyz_vt).float().cuda().requires_grad_(True))

        self.prepare_interpolating_weights(xyz_ft, xyz_vt)

        self.init()

    def training_setup(self, args: Config, scene_scale):
        eps=1e-15 
        betas = (1 - 1 * (1 - 0.9), 1 - 1 * (1 - 0.999))
        decay = 0.001

        self._texture_color_lr = args.texture_color_lr
        self._texture_alpha_lr = args.texture_alpha_lr

        optimizers = {
            'dxyz': Adam([self.dxyz_vt], args.position_lr * scene_scale, betas, eps),
            'scales': Adam([self._scaling], args.scaling_lr, betas, eps),
            'quats': Adam([self._rotation], args.rotation_lr, betas, eps),
            'sh0': Adam([self._sh0], args.color_lr, betas, eps),
            'shN': Adam([self._shN], args.color_lr / 20, betas, eps),
            'texture_color': Adam(self._texture_color, 0, betas, eps),
            'texture_alpha': Adam(self._texture_alpha, 0, betas, eps),

            'dxyz_bs': Adam([self.dxyz_bs], args.position_lr * scene_scale / 10, betas, eps),
            'dscales_bs': Adam([self.scaling_bs], args.scaling_lr / 5, betas, eps),
            'dquats_bs': Adam([self.rotation_bs], args.rotation_lr / 5, betas, eps),
            'dsh0_bs': Adam([self.sh0_bs], args.color_lr / 5, betas, eps),
            'dshN_bs': Adam([self.shN_bs], args.color_lr / 200, betas, eps),

            'encoder_feat_params': AdamW(self.encoder_feat_params.values(), args.encoder_lr, betas, eps, decay),

            'xyz_offset': Adam([self.xyz_offset], args.xyz_offset_lr, betas, eps),
        }

        schedulers = [
            ExponentialLR(optimizers['dxyz'], gamma=0.01 ** (1.0 / args.iterations)),
            ExponentialLR(optimizers['scales'], gamma=0.1 ** (1.0 / args.iterations)),
            ExponentialLR(optimizers['quats'], gamma=0.1 ** (1.0 / args.iterations)),
            ExponentialLR(optimizers['sh0'], gamma=0.1 ** (1.0 / args.iterations)),
            ExponentialLR(optimizers['shN'], gamma=0.1 ** (1.0 / args.iterations)),
            ExponentialLR(optimizers['texture_color'], gamma=0.1 ** (1.0 / args.iterations)),
            ExponentialLR(optimizers['texture_alpha'], gamma=0.1 ** (1.0 / args.iterations)),

            ExponentialLR(optimizers['dxyz_bs'], gamma=0.1 ** (1.0 / args.iterations)),
            ExponentialLR(optimizers['encoder_feat_params'], gamma=0.1 ** (1.0 / args.iterations)),

            ExponentialLR(optimizers['xyz_offset'], gamma=0.1 ** (1.0 / args.iterations)),
        ]

        self.optimizers = optimizers
        self.schedulers = schedulers

    def optimizer_step(self):
        for optimizer in self.optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for scheduler in self.schedulers:
            scheduler.step()
        
        self.cache_dict = {}

    def render(self, cam, scaling_modifier=1.0, background=None):
        means2D = torch.zeros_like(self.get_xyz, dtype=self.get_xyz.dtype, requires_grad=True, device=self.get_xyz.device)
        means2D.retain_grad()

        FoVx = focal2fov(cam['K'][0, 0].item(), cam['width'])
        FoVy = focal2fov(cam['K'][1, 1].item(), cam['height'])
        tanfovx = math.tan(FoVx * 0.5)
        tanfovy = math.tan(FoVy * 0.5)
        proj_mat = getProjectionMatrix(znear=0.1, zfar=100, fovX=FoVx, fovY=FoVy, K=cam['K'],
                                                img_w=cam['width'], img_h=cam['height']).cuda()

        proj_mat = proj_mat.T
        viewmatrix = cam['w2c'].T

        full_proj_transform = (viewmatrix[None].bmm(proj_mat[None]))[0]
        camera_center = viewmatrix.inverse()[3, :3]

        torch.cuda.synchronize()
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)
        iter_start.record()

        scales = self.get_cano_scaling * scaling_modifier
        rots = self.get_Gweights[:, :3, :3].contiguous()
        if self.Rh is not None: rots = self.Rh @ rots
        rots = matrix_to_quaternion(rots)
        rotations = quaternion_multiply(rots, self.get_cano_rotation)

        raster_settings = GaussianRasterizationSettings(
            image_height=cam['height'],
            image_width=cam['width'],
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=background[None],
            scale_modifier=1.0,
            viewmatrix=viewmatrix,
            projmatrix=full_proj_transform,
            sh_degree=self.sh_degree,
            campos=camera_center,
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        textures_color = self.get_texture_color
        textures_alpha = self.get_texture_alpha

        LOD2_color = textures_color[1]
        LOD2_alpha = textures_alpha[1]

        cam_pos = torch.linalg.inv_ex(viewmatrix)[0][:3, 3]
        override_color = self.get_color(cam_pos)

        rendered_image, radii, impact, allmap = rasterizer(
            means3D=self.get_xyz,
            means2D=means2D,
            texture_alpha_LOD1=textures_alpha[0], texture_color_LOD1=textures_color[0],
            texture_alpha_LOD2=LOD2_alpha, texture_color_LOD2=LOD2_color,
            shs=None, #self.get_sh,
            colors_precomp=override_color,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )

        iter_end.record()
        torch.cuda.synchronize()
        run_time = iter_start.elapsed_time(iter_end)

        render_alpha = allmap[1:2]

        # additional regularizations
        # get normal map
        render_normal = allmap[2:5]
        source_normal = -render_normal
        render_normal = (render_normal.permute(1, 2, 0) @ (viewmatrix[:3, :3].T)).permute(2,0,1)

        # get median depth map
        render_depth_median = allmap[5:6]
        render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

        # get expected depth map — detach denominator so 0/0 in empty regions doesn't NaN the gradient
        render_depth_expected = allmap[0:1]
        render_depth_expected = (render_depth_expected / render_alpha.detach().clamp(min=1e-8))
        render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

        # get depth distortion map
        render_dist = allmap[6:7]

        # psedo surface attributes
        depth_ratio = 1.0
        surf_depth = render_depth_expected * (1 - depth_ratio) + (depth_ratio) * render_depth_median

        # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
        # detach surf_depth: surf_normal is a fixed pseudo-surface target; gradients should not flow
        # back through depth_to_normal (F.normalize backward at zero-depth pixels gives ~1e12 gradient)
        surf_normal = depth_to_normal(viewmatrix, tanfovx, tanfovy, [cam['width'], cam['height']], surf_depth.detach())
        surf_normal = surf_normal.permute(2, 0, 1)
        surf_normal = torch.nan_to_num(surf_normal, 0, 0)
        # remember to multiply with accum_alpha since render_normal is unnormalized.
        surf_normal = surf_normal * (render_alpha).detach()
        
        info = {
            "render_normal": render_normal,
            "source_normal": source_normal,
            "render_dist": render_dist,
            "surf_depth": surf_depth,
            "surf_normal": surf_normal,
            "impact": impact,
            "run_time": run_time,
        }
        #info = {}

        rendered_image = rendered_image.permute(1, 2, 0)
        render_alpha = render_alpha.permute(1, 2, 0)
        return rendered_image, render_alpha, info

    def init_body(self):
        # Rots = batch_rodrigues(smpl.smpl_bigpose.reshape(-1,3)).cuda()
        # Ac = batch_rigid_transform(Rots[None], self.t_joints[None], self.joint_parents)[1][0]
        Ac = rigid_transform_tensor(smpl.smpl_bigpose, self.t_joints, self.joint_parents).cpu()
        self.Ac_inv = torch.linalg.inv(Ac)
        self.reset_pose()

    def reset_pose(self):
        self.Rh = torch.eye(3, dtype=torch.float32, device='cpu')
        self.Th = torch.zeros(3, dtype=torch.float32, device='cpu')
        self.smpl_poses = smpl.smpl_tpose.cpu()

    @property
    def smpl_poses(self):
        return self._smpl_poses
    
    @smpl_poses.setter
    def smpl_poses(self, value):
        self.cache_dict = {}
        self._smpl_poses = value.cpu()
        self.smpl_poses_cuda = value.cuda(non_blocking=True)

    @property
    def Rh(self):
        return self._Rh
    
    @Rh.setter
    def Rh(self, value):
        if np.allclose(value.cpu().numpy(), np.eye(3), atol=1e-5):
            self._Rh = None
        else:
            self._Rh = value.cuda(non_blocking=True)

    @property
    def Th(self):
        return self._Th
    
    @Th.setter
    def Th(self, value):
        self._Th = value.cuda(non_blocking=True)

    def prepare_interpolating_weights(self, xyz_ft, xyz_vt):
        self.xyz_vt = xyz_vt
        self.xyz_ft = xyz_ft

        dists, idxs, _ = knn_points(
            p1=self._xyz[None],
            p2=xyz_vt[None],
            K=3,
        )
        nbr_gs = idxs[0]
        nbr_gs_invdist = 1 / torch.sqrt(dists[0])
        nbr_gs_wght = nbr_gs_invdist / torch.sum(nbr_gs_invdist, dim=-1, keepdim=True)

        _, idxs, _ = knn_points(
            p1=xyz_vt[None],
            p2=xyz_vt[None],
            K=7,
        )
        nbr_vt = idxs[0]

        self.nbr_gs = nbr_gs
        self.nbr_gs_invdist = nbr_gs_invdist
        self.nbr_vt = nbr_vt

        dists, idxs, _ = knn_points(
            p1=self._xyz[None],
            p2=xyz_ft[None],
            K=3,
        )
        nbr_gs = idxs[0]
        nbr_gs_invdist = 1 / torch.sqrt(dists[0])
        nbr_gs_wght = nbr_gs_invdist / torch.sum(nbr_gs_invdist, dim=-1, keepdim=True)
        self.nbr_gsft = nbr_gs
        self.nbr_gsft_wght = nbr_gs_wght

        dists, idxs, _ = knn_points(
            p1=self.xyz_vt[None],
            p2=xyz_ft[None],
            K=3,
        )
        nbr_gs = idxs[0]
        nbr_gs_invdist = 1 / torch.sqrt(dists[0])
        nbr_gs_wght = nbr_gs_invdist / torch.sum(nbr_gs_invdist, dim=-1, keepdim=True)
        self.nbr_vtft = nbr_gs
        self.nbr_vtft_wght = nbr_gs_wght
