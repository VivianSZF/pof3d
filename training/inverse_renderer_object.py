#modified from Unsup3D(https://github.com/elliottwu/unsup3d)

import torch
import torch.nn as nn
import torchvision
from .volumetric_rendering.inverse_renderer import Renderer
import math

EPS = 1e-7


class InverseRendererObject(nn.Module):
    def __init__(self, 
                 resolution=64,
                 min_depth=2.25,
                 max_depth=3.3,
                 rot_center_depth=None,
                 border_depth=None,
                 min_amb_light=0,
                 max_amb_light=1,
                 min_diff_light=0,
                 max_diff_light=1,
                 xyz_rotation_range=60,
                 xy_translation_range=0.1,
                 z_translation_range=0,
                 use_conf_map=True,
                 fov=18.837,
                 tex_cube_size=2,
                 renderer_min_depth=0.1,
                 renderer_max_depth=10.,
                 device=None
                 ):
        super().__init__()
        self.resolution = resolution
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.border_depth = border_depth if border_depth else (0.7*self.max_depth + 0.3*self.min_depth)
        self.rot_center_depth = rot_center_depth if rot_center_depth else (min_depth+max_depth)/2
        self.min_amb_light = min_amb_light
        self.max_amb_light = max_amb_light
        self.min_diff_light = min_diff_light
        self.max_diff_light = max_diff_light
        self.xyz_rotation_range = xyz_rotation_range
        self.xy_translation_range = xy_translation_range
        self.z_translation_range = z_translation_range
        self.use_conf_map = use_conf_map
        self.renderer = Renderer(resolution=resolution,
                               min_depth=min_depth,
                               max_depth=max_depth,
                               rot_center_depth=self.rot_center_depth,
                               fov=fov,
                               tex_cube_size=tex_cube_size,
                               renderer_min_depth=renderer_min_depth,
                               renderer_max_depth=renderer_max_depth,
                               device=device                          
                               )
        self.netD = EDDeconv(cin=3, cout=1, nf=64, zdim=256, activation=None)
        self.netA = EDDeconv(cin=3, cout=3, nf=64, zdim=256)
        self.netL = Encoder(cin=3, cout=4, nf=32)
        self.netV = Encoder(cin=3, cout=6, nf=32)
        if self.use_conf_map:
            self.netC = ConfNet(cin=3, cout=2, nf=64, zdim=128)


    def depth_rescaler(self, d):
        return (1+d)/2 *self.max_depth + (1-d)/2 *self.min_depth
    
    def amb_light_rescaler(self, x):
        return (1+x)/2 *self.max_amb_light + (1-x)/2 *self.min_amb_light
    
    def diff_light_rescaler(self, x):
        return (1+x)/2 *self.max_diff_light + (1-x)/2 *self.min_diff_light

    def forward(self, input):
        """Feedforward once."""
        self.input_im = input# *2.-1.
        b, c, h, w = self.input_im.shape

        ## predict canonical depth
        self.canon_depth_raw = self.netD(self.input_im).squeeze(1)  # BxHxW
        self.canon_depth = self.canon_depth_raw - self.canon_depth_raw.view(b,-1).mean(1).view(b,1,1)
        self.canon_depth = self.canon_depth.tanh()
        self.canon_depth = self.depth_rescaler(self.canon_depth)

        ## optional depth smoothness loss (only used in synthetic car experiments)
        # self.loss_depth_sm = ((self.canon_depth[:,:-1,:] - self.canon_depth[:,1:,:]) /(self.max_depth-self.min_depth)).abs().mean()
        # self.loss_depth_sm += ((self.canon_depth[:,:,:-1] - self.canon_depth[:,:,1:]) /(self.max_depth-self.min_depth)).abs().mean()

        ## clamp border depth
        depth_border = torch.zeros(1,h,w-4).to(self.input_im.device)
        depth_border = nn.functional.pad(depth_border, (2,2), mode='constant', value=1)
        self.canon_depth = self.canon_depth*(1-depth_border) + depth_border *self.border_depth
        self.canon_depth = torch.cat([self.canon_depth, self.canon_depth.flip(2)], 0)  # flip

        ## predict canonical albedo
        self.canon_albedo = self.netA(self.input_im)  # Bx3xHxW
        self.canon_albedo = torch.cat([self.canon_albedo, self.canon_albedo.flip(3)], 0)  # flip

        ## predict confidence map
        if self.use_conf_map:
            conf_sigma_l1, conf_sigma_percl = self.netC(self.input_im)  # Bx2xHxW
            self.conf_sigma_l1 = conf_sigma_l1[:,:1]
            self.conf_sigma_l1_flip = conf_sigma_l1[:,1:]
            self.conf_sigma_percl = conf_sigma_percl[:,:1]
            self.conf_sigma_percl_flip = conf_sigma_percl[:,1:]
        else:
            self.conf_sigma_l1 = None
            self.conf_sigma_l1_flip = None
            self.conf_sigma_percl = None
            self.conf_sigma_percl_flip = None

        ## predict lighting
        canon_light = self.netL(self.input_im).repeat(2,1)  # Bx4
        self.canon_light_a = self.amb_light_rescaler(canon_light[:,:1])  # ambience term
        self.canon_light_b = self.diff_light_rescaler(canon_light[:,1:2])  # diffuse term
        canon_light_dxy = canon_light[:,2:]
        self.canon_light_d = torch.cat([canon_light_dxy, torch.ones(b*2,1).to(self.input_im.device)], 1)
        self.canon_light_d = self.canon_light_d / ((self.canon_light_d**2).sum(1, keepdim=True))**0.5  # diffuse light direction

        ## shading
        self.canon_normal = self.renderer.get_normal_from_depth(self.canon_depth)
        self.canon_diffuse_shading = (self.canon_normal * self.canon_light_d.view(-1,1,1,3)).sum(3).clamp(min=0).unsqueeze(1)
        canon_shading = self.canon_light_a.view(-1,1,1,1) + self.canon_light_b.view(-1,1,1,1)*self.canon_diffuse_shading
        self.canon_im = (self.canon_albedo/2+0.5) * canon_shading *2-1

        ## predict viewpoint transformation
        self.view = self.netV(self.input_im).repeat(2,1)
        self.view = torch.cat([
            self.view[:,:3] *math.pi/180 *self.xyz_rotation_range,
            self.view[:,3:5] *self.xy_translation_range,
            self.view[:,5:] *self.z_translation_range], 1)

        ## reconstruct input view
        self.renderer.set_transform_matrices(self.view)
        self.recon_depth = self.renderer.warp_canon_depth(self.canon_depth)
        self.recon_normal = self.renderer.get_normal_from_depth(self.recon_depth)
        grid_2d_from_canon = self.renderer.get_inv_warped_2d_grid(self.recon_depth)
        self.recon_im = nn.functional.grid_sample(self.canon_im, grid_2d_from_canon, mode='bilinear')

        margin = (self.max_depth - self.min_depth) /2
        recon_im_mask = (self.recon_depth < self.max_depth+margin).float()  # invalid border pixels have been clamped at max_depth+margin
        recon_im_mask_both = recon_im_mask[:b] * recon_im_mask[b:]  # both original and flip reconstruction
        recon_im_mask_both = recon_im_mask_both.repeat(2,1,1).unsqueeze(1).detach()
        self.recon_im = self.recon_im * recon_im_mask_both

        ## render symmetry axis
        canon_sym_axis = torch.zeros(h, w).to(self.input_im.device)
        canon_sym_axis[:, w//2-1:w//2+1] = 1
        self.recon_sym_axis = nn.functional.grid_sample(canon_sym_axis.repeat(b*2,1,1,1), grid_2d_from_canon, mode='bilinear')
        self.recon_sym_axis = self.recon_sym_axis * recon_im_mask_both
        green = torch.FloatTensor([-1,1,-1]).to(self.input_im.device).view(1,3,1,1)
        self.input_im_symline = (0.5*self.recon_sym_axis) *green + (1-0.5*self.recon_sym_axis) *self.input_im.repeat(2,1,1,1)

        results = {
            'recon_im': self.recon_im,
            'recon_im_mask_both': recon_im_mask_both,
            'canon_depth': self.canon_depth,
            'recon_depth': self.recon_depth,
            'conf_sigma_l1': self.conf_sigma_l1,
            'conf_sigma_l1_flip': self.conf_sigma_l1_flip,
            'conf_sigma_percl': self.conf_sigma_percl,
            'conf_sigma_percl_flip': self.conf_sigma_percl_flip,
            'canon_im': self.canon_im,
            'canon_albedo': self.canon_albedo,
            'canon_diffuse_shading': self.canon_diffuse_shading,
            'recon_normal': self.recon_normal
        }

        return results




class Encoder(nn.Module):
    def __init__(self, cin, cout, nf=64, activation=nn.Tanh):
        super(Encoder, self).__init__()
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, nf*8, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, cout, kernel_size=1, stride=1, padding=0, bias=False)]
        if activation is not None:
            network += [activation()]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input).reshape(input.size(0),-1)


class EDDeconv(nn.Module):
    def __init__(self, cin, cout, zdim=128, nf=64, activation=nn.Tanh):
        super(EDDeconv, self).__init__()
        ## downsampling
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.GroupNorm(16, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(16*2, nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.GroupNorm(16*4, nf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*8, zdim, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            nn.ReLU(inplace=True)]
        ## upsampling
        network += [
            nn.ConvTranspose2d(zdim, nf*8, kernel_size=4, stride=1, padding=0, bias=False),  # 1x1 -> 4x4
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*8, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 4x4 -> 8x8
            nn.GroupNorm(16*4, nf*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16*4, nf*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*4, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 16x16
            nn.GroupNorm(16*2, nf*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16*2, nf*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 32x32
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 32x32 -> 64x64
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, cout, kernel_size=5, stride=1, padding=2, bias=False)]
        if activation is not None:
            network += [activation()]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input)


class ConfNet(nn.Module):
    def __init__(self, cin, cout, zdim=128, nf=64):
        super(ConfNet, self).__init__()
        ## downsampling
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.GroupNorm(16, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(16*2, nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.GroupNorm(16*4, nf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*8, zdim, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            nn.ReLU(inplace=True)]
        ## upsampling
        network += [
            nn.ConvTranspose2d(zdim, nf*8, kernel_size=4, padding=0, bias=False),  # 1x1 -> 4x4
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*8, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 4x4 -> 8x8
            nn.GroupNorm(16*4, nf*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*4, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 16x16
            nn.GroupNorm(16*2, nf*2),
            nn.ReLU(inplace=True)]
        self.network = nn.Sequential(*network)

        out_net1 = [
            nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 32x32
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 64x64
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, 2, kernel_size=5, stride=1, padding=2, bias=False),  # 64x64
            nn.Softplus()]
        self.out_net1 = nn.Sequential(*out_net1)

        out_net2 = [nn.Conv2d(nf*2, 2, kernel_size=3, stride=1, padding=1, bias=False),  # 16x16
                    nn.Softplus()]
        self.out_net2 = nn.Sequential(*out_net2)

    def forward(self, input):
        out = self.network(input)
        return self.out_net1(out), self.out_net2(out)
