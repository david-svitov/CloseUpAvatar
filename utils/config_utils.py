
from dataclasses import dataclass

@dataclass
class ConfigTest:
    cam_ids = [0]
    num_frame = 500
    begin_ith_frame = 0
    frame_interval = 1
    image_scaling = 1

@dataclass
class Config:
    seed = 0
    ip: str
    port: int
    out_dir: str
    data_dir: str
    detect_anomaly = False
    test_iterations = []
    save_iterations = []
    checkpoint_iterations = []

    smpl_pkl_path = './smpl_model/smplx/SMPLX_NEUTRAL.npz'

    background = [1, 1, 1]
    random_background = True

    # dataset
    train_cam_ids = []
    num_train_frame = 100
    begin_ith_frame = 0
    frame_interval = 1
    image_scaling = 1
    data_in_memory = False

    test: ConfigTest

    # optimization
    iterations = 800_000
    position_lr = 0.00016
    opacity_lr = 0.0005
    scaling_lr = 0.0005
    rotation_lr = 0.0005
    color_lr = 0.0005
    xyz_offset_lr = 0.001
    encoder_lr = 0.0005
    texture_color_lr = 0.0005
    texture_alpha_lr = 0.0005

    iteration_sh_degree = 250000

    # loss
    lambda_lpips = 0.1
    iteration_lpips = 6000
    iteration_lpips_random_patch = 300000
    lambda_scaling = 0.1
    scaling_threshold = 0.01
    lambda_dxyz_smooth = 0.1
    lambda_normal = 0.0
    lambda_dist = 0.0
    lambda_texture_value = 0.0001
    lambda_alpha_value = 0.0001
    max_impact_threshold = 100
    
    init_num_gs = 200_000

    # anchor points and control points
    num_verts = 10000
    num_features = 300

    # iteration to start optimize the basis
    iteration_dxyz_basis = 2000
    iteration_gsparam_basis = 2000
