

import math
import numpy as np
import torch


def rand_point_on_mesh(vert, face, pts_num=7000, init_factor=5):
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vert)
    mesh.triangles = o3d.utility.Vector3iVector(face)
    point_cloud = mesh.sample_points_poisson_disk(number_of_points=pts_num, init_factor=init_factor)
    point_cloud = np.asarray(point_cloud.points)
    return point_cloud


def polar_decomposition_newton_schulz(A, iteration=4):
    n, m = A.shape[-2:]
    if n != m:
        raise ValueError("Matrix must be square")

    Q = A.clone()
    for _ in range(iteration):
        Q_t = Q.transpose(-2, -1)
        Q = 0.5 * (Q + torch.linalg.inv_ex(Q_t)[0])
    return Q

def getProjectionMatrix(znear, zfar, fovX, fovY, K = None, img_h = None, img_w = None):
    if K is None:
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))
        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right
    else:
        near_fx = znear / K[0, 0]
        near_fy = znear / K[1, 1]

        left = - (img_w - K[0, 2]) * near_fx
        right = K[0, 2] * near_fx
        bottom = (K[1, 2] - img_h) * near_fy
        top = K[1, 2] * near_fy

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def depths_to_points(viewmatrix, tanfovx, tanfovy, render_size, depthmap):
    c2w = (viewmatrix.T).inverse()
    W, H = render_size[0], render_size[1]
    fx = W / (2 * tanfovx)
    fy = H / (2 * tanfovy)
    intrins = torch.tensor(
        [[fx, 0., W/2.],
        [0., fy, H/2.],
        [0., 0., 1.0]]
    ).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

def depth_to_normal(viewmatrix, tanfovx, tanfovy, render_size, depth):
    """
        view: view camera
        depth: depthmap
    """
    points = depths_to_points(viewmatrix, tanfovx, tanfovy, render_size, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output
