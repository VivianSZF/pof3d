import torch
from torch.nn.functional import grid_sample
import numpy as np


def warp_img1_to_img0(depth0, intrinsic, c2w0, w2c1, img1):
    N, _, H, W = img1.shape
    
    points = unproject(depth0, intrinsic).reshape(N, -1, 3)
    points[..., 2] *= -1
    points = torch.bmm(points, torch.transpose(c2w0[:, :3, :3], 1, 2)) + c2w0[:, :3, 3].unsqueeze(1)

    xy1 = project(points, intrinsic, w2c1[:, :3]).reshape(N, H, W, 2)
    grid_ = 2* xy1 / H -1 
    mask = 1 - ((grid_[:,:,:,0] < -1) | (grid_[:,:,:,0] > 1) | (grid_[:,:,:,1] < -1) | (grid_[:,:,:,1] > 1)).float() # indicate the valie region
    img0_gridfrom_img1 = grid_sample(img1, grid_, align_corners=False)

    return img0_gridfrom_img1, mask.unsqueeze(1)


def unproject(depth_map, K):
    """
    depth_map: N, h, w
    K: 3, 3
    """
    depth_map = depth_map.squeeze(1)
    N, H, W = depth_map.shape
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W))
    y = y.to(depth_map.device).unsqueeze(0).repeat(N,1,1)
    x = x.to(depth_map.device).unsqueeze(0).repeat(N,1,1)
    xy_map = torch.stack([x, y], axis=3) * depth_map[..., None]
    xyz_map = torch.cat([xy_map, depth_map[..., None]], axis=-1)
    xyz = xyz_map.view(-1, 3)
    xyz = torch.matmul(xyz, torch.transpose(torch.inverse(K[0]), 0, 1))
    xyz_map = xyz.view(N, H, W, 3)
    return xyz_map

def project(xyz, K, RT):
    """
    xyz: [N, HW, 3]
    K: [N, 3, 3]
    RT: [N, 3, 4]

    Output:
    xy: [N, HW. 2]
    """
    xyz = torch.bmm(xyz, torch.transpose(RT[:, :, :3], 1, 2)) + torch.transpose(RT[:, :, 3:], 1, 2)
    xyz[:, :, 2] *= -1
    xyz = torch.bmm(xyz, torch.transpose(K, 1, 2))
    xy = xyz[:, :, :2] / xyz[:, :, 2:]
    return xy

def get_K(H, W, fov):
    fx = W / (2 * np.tan(np.pi / 360 * fov))
    fy = -H / (2 * np.tan(np.pi / 360 * fov))
    cx = W / 2
    cy = H / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return K
