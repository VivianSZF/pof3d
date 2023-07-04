# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Model for PoF3D

import torch
from torch_utils import persistence
from training.networks_stylegan2 import Generator as StyleGAN2Backbone
from training.volumetric_rendering.renderer import ImportanceRenderer
from training.volumetric_rendering.ray_sampler import RaySampler, depth2pts_outside
from training.volumetric_rendering.ray_marcher import MipRayMarcher2
from camera_utils import LookAtPose
import dnnlib
import math
import torch.nn.functional as F
from einops import repeat, rearrange

@persistence.persistent_class
class TriPlaneGeneratorPose(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        sr_num_fp16_res     = 0,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        sr_kwargs = {},
        bg_kwargs = None,
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.renderer = ImportanceRenderer(wrong=rendering_kwargs['wrong'])
        self.ray_sampler = RaySampler()
        self.ray_marcher = MipRayMarcher2()
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32*3, mapping_kwargs=mapping_kwargs, bg_kwargs=bg_kwargs, **synthesis_kwargs)
        self.bg_kwargs = bg_kwargs
        if rendering_kwargs['superresolution_module'] is None:
            self.superresolution = None
        else:
            self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32})
        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs
        self.clamp_h = rendering_kwargs.get('clamp_h', None)
        self.clamp_v = rendering_kwargs.get('clamp_v', None)
        self.h_mean = rendering_kwargs.get('h_mean', math.pi/2)
        self.r_mean = rendering_kwargs.get('r_mean', 2.7)
        self.fov_mean = rendering_kwargs.get('fov_mean', 20)
        self.detach_w_superres = rendering_kwargs.get('detach_w_superres', False)
        self.learn_fov = rendering_kwargs.get('learn_fov', False)

    
        self._last_planes = None
    
    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, detach_w=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
                c = torch.zeros_like(c)
        mapping_results = self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, 
                            update_emas=update_emas, detach_w=detach_w, clamp_h=self.clamp_h, clamp_v=self.clamp_v, h_mean=self.h_mean, r_mean=self.r_mean, fov_mean=self.fov_mean)
        ws, cam_pose = mapping_results['ws'], mapping_results['cam_params']
        w_before = mapping_results['before_repeat_w']
        if cam_pose.shape[1] == 2 or (cam_pose.shape[1] == 3 and self.learn_fov):
            cam2world_matrix = LookAtPose.sample(cam_pose[:,0:1], cam_pose[:,1:2], torch.tensor(self.rendering_kwargs['avg_camera_pivot'], device=ws.device), 
                                                            radius=self.rendering_kwargs['avg_camera_radius'], device=ws.device)
        else:
            assert cam_pose.shape[1]==3 and not self.learn_fov
            cam2world_matrix = LookAtPose.sample(cam_pose[:,0:1], cam_pose[:,1:2], torch.tensor(self.rendering_kwargs['avg_camera_pivot'], device=ws.device), 
                                                            radius=cam_pose[:, 2:3], device=ws.device)
        return {'ws': ws, 
                'before_repeat_w': w_before,
                'c2w': cam2world_matrix,
                'cam_pose': cam_pose}

    def synthesis(self, ws, c, cam2world_matrix=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, flip=False, flip_type='flip', fov=None, **synthesis_kwargs):
        
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)
        cam2world_matrix_2 = None
        
        if flip:
            if flip_type=='flip':
            #(2,0) (0,1) (0,2) (0,3)
                cam2world_matrix_2 = cam2world_matrix.clone()
                cam2world_matrix_2[:, 2, 0]*=-1
                cam2world_matrix_2[:, 0, 1]*=-1
                cam2world_matrix_2[:, 0, 2]*=-1
                cam2world_matrix_2[:, 0, 3]*=-1
                ray_origins_flip, ray_directions_flip = self.ray_sampler(cam2world_matrix_2, intrinsics, neural_rendering_resolution)
                ray_origins = torch.cat((ray_origins, ray_origins_flip), dim=0)
                ray_directions = torch.cat((ray_directions, ray_directions_flip), dim=0)
            elif flip_type=='roll':
                cam2world_matrix_2 = torch.roll(cam2world_matrix.clone(), 1, 0)
                ray_origins_roll, ray_directions_roll = self.ray_sampler(cam2world_matrix_2, intrinsics, neural_rendering_resolution)
                ray_origins = torch.cat((ray_origins, ray_origins_roll), dim=0)
                ray_directions = torch.cat((ray_directions, ray_directions_roll), dim=0)


        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws[:,:self.backbone.synthesis.num_ws], update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

        # Perform volume rendering
        feature_samples, depth_samples, weights_samples, bg_lambda = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last

        # background rendering
        if self.bg_kwargs is not None:
            di = torch.linspace(-1., 0., steps=self.bg_kwargs.get('num_bg_pts', 4)).to(planes.device)
            di = repeat(di, 's -> b n s', b=N, n=M) * self.bg_kwargs.get('bg_start', 0.5)
            di = add_noise_to_interval(di)
            n_steps  = di.shape[-1]
            bg_pts, _ = depth2pts_outside(ray_origins.unsqueeze(-2).expand(N, M, n_steps, 3), ray_directions.unsqueeze(-2).expand(N, M, n_steps, 3), -di)
            bg_pts = bg_pts.reshape(N, -1, 4)
            bg_shape = [N, neural_rendering_resolution, neural_rendering_resolution, n_steps]
            feat, sigma_raw = self.backbone.bg_nerf(bg_pts, ray_directions, ws=ws[:,-self.backbone.bg_nerf.num_ws:], shape=bg_shape)
            feat      = rearrange(feat, 'b (n s) d -> b n s d', s=n_steps)
            sigma_raw = rearrange(sigma_raw, 'b (n s) d -> b n s d', s=n_steps)
            feat_final, _, _, _ = self.ray_marcher(feat, sigma_raw, di.unsqueeze(-1), self.rendering_kwargs)
            feat_final = bg_lambda * feat_final
            feature_samples = feature_samples + feat_final



        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        if depth_image.size(2) != 64:
            scale_factor = 64 / depth_image.size(2)
            input_depth = F.interpolate(depth_image.clone(), scale_factor=scale_factor, mode='bilinear').squeeze(1)
        else:
            input_depth = depth_image.clone().squeeze(1)
        normal_map = get_normal_from_depth(input_depth).permute(0,3,1,2)

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        if self.superresolution is not None:
            if self.detach_w_superres:
                sr_image = self.superresolution(rgb_image, feature_image, ws.detach(), noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
            else:
                sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        else:
            sr_image = rgb_image

        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image, 'normal_map': normal_map, 'c2w': cam2world_matrix, 'c2w_flip': cam2world_matrix_2, 'intrinsic': intrinsics}
    
    def sample(self, coordinates, directions, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        if z.ndim == 2:
            ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)['ws']
        else:
            ws = z
        planes = self.backbone.synthesis(ws[:,:self.backbone.synthesis.num_ws], update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def sample_mixed(self, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes = self.backbone.synthesis(ws[:,:self.backbone.synthesis.num_ws], update_emas = update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, flip=False, **synthesis_kwargs):
        # Render a batch of generated images.

        mapping_results = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        if self.learn_fov:
            return self.synthesis(mapping_results['ws'], c, cam2world_matrix=mapping_results['c2w'], update_emas=update_emas, 
                    neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, 
                    flip=flip, fov=mapping_results['cam_pose'][:,2:3], **synthesis_kwargs)
        else:
            return self.synthesis(mapping_results['ws'], c, cam2world_matrix=mapping_results['c2w'], update_emas=update_emas, 
                    neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, 
                    flip=flip, **synthesis_kwargs)

def add_noise_to_interval(di):
    di_mid  = .5 * (di[..., 1:] + di[..., :-1])
    di_high = torch.cat([di_mid, di[..., -1:]], dim=-1)
    di_low  = torch.cat([di[..., :1], di_mid], dim=-1)
    noise   = torch.rand_like(di_low)
    ti      = di_low + (di_high - di_low) * noise
    return ti


def get_grid(b, H, W, normalize=True):
    if normalize:
        h_range = torch.linspace(-1,1,H)
        w_range = torch.linspace(-1,1,W)
    else:
        h_range = torch.arange(0,H)
        w_range = torch.arange(0,W)
    grid = torch.stack(torch.meshgrid([h_range, w_range]), -1).repeat(b,1,1,1).flip(3).float() # flip h,w to x,y
    return grid

def depth_to_3d_grid(depth, inv_K):
        b, h, w = depth.shape
        grid_2d = get_grid(b, h, w, normalize=True).to(depth.device)  # Nxhxwx2
        depth = depth.unsqueeze(-1)
        grid_3d = torch.cat((grid_2d, torch.ones_like(depth)), dim=3)
        grid_3d = grid_3d.matmul(inv_K.to(depth.device).transpose(2,1)) * depth
        return grid_3d
    
def get_normal_from_depth(depth, res=2):
    fov=18.837
    R = [[[1.,0.,0.],
            [0.,1.,0.],
            [0.,0.,1.]]]
    R = torch.FloatTensor(R).to(depth.device)
    t = torch.zeros(1,3, dtype=torch.float32).to(depth.device)
    fx = 1/(math.tan(fov/2 *math.pi/180)) # TODO: Check.
    fy = 1/(math.tan(fov/2 *math.pi/180))
    cx = 0
    cy = 0
    K = [[fx, 0., cx],
            [0., fy, cy],
            [0., 0., 1.]]
    K = torch.FloatTensor(K).to(depth.device)
    inv_K = torch.inverse(K).unsqueeze(0)
    b, h, w = depth.shape
    grid_3d = depth_to_3d_grid(depth, inv_K)

    tu = grid_3d[:,1:-1,2:] - grid_3d[:,1:-1,:-2]
    tv = grid_3d[:,2:,1:-1] - grid_3d[:,:-2,1:-1]
    normal = tu.cross(tv, dim=3)

    zero = torch.FloatTensor([0,0,1]).to(depth.device)
    normal = torch.cat([zero.repeat(b,h-2,1,1), normal, zero.repeat(b,h-2,1,1)], 2)
    normal = torch.cat([zero.repeat(b,1,w,1), normal, zero.repeat(b,1,w,1)], 1)
    normal = normal / (torch.norm(normal, p=2, dim=3, keepdim=True) + 1e-7)
    return normal

from training.networks_stylegan2 import FullyConnectedLayer

class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
        
    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}


def get_density(self, sigma_raw, sigma_type='softplus'):
    if sigma_type == 'relu':
        sigma_raw = sigma_raw + torch.randn_like(sigma_raw)
        sigma = F.relu(sigma_raw)
    elif sigma_type == 'softplus':  # https://arxiv.org/pdf/2111.11215.pdf
        sigma = F.softplus(sigma_raw - 1)       # 1 is the shifted bias.
    elif sigma_type == 'exp_truncated':    # density in the log-space
        sigma = torch.exp(5 - F.relu(5 - (sigma_raw - 1)))  # up-bound = 5, also shifted by 1
    else:
        sigma = sigma_raw
    return sigma