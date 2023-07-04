# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
import math
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from training.dual_discriminator import filtered_resizing
from training.loss_utils import warp_img1_to_img0, get_K
from camera_utils import LookAtPose
import torch.nn as nn
import torchvision

import os
import cv2
import random
import torch.nn.functional as F
EPS = 1e-7
#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2LossPose(Loss):
    def __init__(self, device, G, D, IR=None, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, 
                    blur_init_sigma=0, blur_fade_kimg=0, r1_gamma_init=0, r1_gamma_fade_kimg=0, neural_rendering_resolution_initial=64, neural_rendering_resolution_final=None, 
                    neural_rendering_resolution_fade_kimg=0, gpc_reg_fade_kimg=1000, gpc_reg_prob=None, dual_discrimination=False, filter_mode='antialiased',
                    min_depth=2.25, max_depth=3.3, lam_flip=0.5, lam_perc=1.0, lam_depth_sm=0.0, normal_start_step=20, normal_end_step=60, normal_gamma_stop=2000,
                    normal_gamma_start=0.0, normal_gamma_end=0.1, geod=False, detach_pose=False, detach_w=False, geod_interval=1,
                    geod_interval_start=0.0):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.IR                 = IR
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.r1_gamma_init      = r1_gamma_init
        self.r1_gamma_fade_kimg = r1_gamma_fade_kimg
        self.neural_rendering_resolution_initial = neural_rendering_resolution_initial
        self.neural_rendering_resolution_final = neural_rendering_resolution_final
        self.neural_rendering_resolution_fade_kimg = neural_rendering_resolution_fade_kimg
        self.gpc_reg_fade_kimg = gpc_reg_fade_kimg
        self.gpc_reg_prob = gpc_reg_prob
        self.dual_discrimination = dual_discrimination
        self.filter_mode = filter_mode
        self.resample_filter = upfirdn2d.setup_filter([1,3,3,1], device=device)
        self.blur_raw_target = True
        self.dis_cam_weight = self.G.rendering_kwargs.get('dis_cam_weight', 0)
        self.pose_gd_weight = self.G.rendering_kwargs.get('pose_gd_weight', 0)
        self.PerceptualLoss = PerceptualLoss(requires_grad=False).to(device=device)
        self.geod = geod
        self.min_depth         = min_depth
        self.max_depth         = max_depth
        self.lam_flip          = lam_flip
        self.lam_perc          = lam_perc
        self.lam_depth_sm      = lam_depth_sm
        self.normal_gamma_start= normal_gamma_start
        self.normal_gamma_end  = normal_gamma_end
        self.normal_start_step = normal_start_step
        self.normal_end_step   = normal_end_step
        self.normal_gamma_stop = normal_gamma_stop
        self.detach_pose       = detach_pose #whether to detach the pose during training(for flipped image for rotated image)
        self.detach_w          = detach_w
        self.geod_interval     = geod_interval
        self.geod_interval_start = geod_interval_start
        self.clamp_h = self.G.rendering_kwargs.get('clamp_h', None)
        self.clamp_v = self.G.rendering_kwargs.get('clamp_v', None)
        self.rnd_pose_ford = self.G.rendering_kwargs.get('rnd_pose_ford', False)
        self.learn_fov = self.G.rendering_kwargs.get('learn_fov', False)

        assert self.gpc_reg_prob is None or (0 <= self.gpc_reg_prob <= 1)

    def run_G(self, z, c, swapping_prob, neural_rendering_resolution, update_emas=False, flip=False, flip_type='flip', cam_flip=False, detach_pose=False):

        mapping_results = self.G.mapping(z, c, update_emas=update_emas)
        ws = mapping_results['ws']
        cam2world_matrix = mapping_results['c2w']
        cam_pose = mapping_results['cam_pose']
        if cam_flip:
            # flip the camera outside, not in the synthesis function, to avoid the minibatch std problem
            assert flip==False
            if flip_type=='flip_both':
                if random.random()<0.5:
                    c_new = cam2world_matrix.clone()
                    c_new[:, 2, 0]*=-1
                    c_new[:, 0, 1]*=-1
                    c_new[:, 0, 2]*=-1
                    c_new[:, 0, 3]*=-1
                else:
                    c_new = cam2world_matrix.clone()
                    c_new[:, 2, 1]*=-1
                    c_new[:, 0, 1]*=-1
                    c_new[:, 1, 0]*=-1
                    c_new[:, 1, 2]*=-1
                    c_new[:, 1, 3]*=-1
            elif flip_type=='flip_both_shapenet':
                if random.random()<0.5:
                    c_new = cam2world_matrix.clone()
                    c_new[:, 0, 0]*=-1
                    c_new[:, 2, 1]*=-1
                    c_new[:, 2, 2]*=-1
                    c_new[:, 2, 3]*=-1
                else:
                    c_new = cam2world_matrix.clone()
                    c_new[:, 2, 1]*=-1
                    c_new[:, 0, 1]*=-1
                    c_new[:, 1, 0]*=-1
                    c_new[:, 1, 2]*=-1
                    c_new[:, 1, 3]*=-1
        else:
            if swapping_prob is not None:
                assert swapping_prob is None or swapping_prob==0
                c_swapped = torch.roll(cam2world_matrix.clone(), 1, 0)
                a = torch.rand((cam2world_matrix.shape[0], 1), device=cam2world_matrix.device) < swapping_prob
                c_new = torch.where(a, c_swapped.reshape(cam2world_matrix.shape[0], 16), cam2world_matrix.reshape(cam2world_matrix.shape[0], 16)).reshape(cam2world_matrix.shape[0], 4,4)
            else:
                c_new = cam2world_matrix
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)['ws'][:, cutoff:]
        if detach_pose:
            c_new = c_new.detach()
        gen_output = self.G.synthesis(ws, c, cam2world_matrix=c_new, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas, flip=flip, flip_type=flip_type, fov=cam_pose[:,2:3])
        return {'gen_output':gen_output, 
                'ws': ws, 
                'cam': c_new,
                'cam_pose': cam_pose}

    def run_D(self, img, c, blur_sigma=0, blur_sigma_raw=0, update_emas=False, cam_pred=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())

        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(torch.cat([img['image'],
                                                    torch.nn.functional.interpolate(img['image_raw'], size=img['image'].shape[2:], mode='bilinear')],
                                                    dim=1))
            img['image'] = augmented_pair[:, :img['image'].shape[1]]
            img['image_raw'] = torch.nn.functional.interpolate(augmented_pair[:, img['image'].shape[1]:], size=img['image_raw'].shape[2:], mode='bilinear')

        results = self.D(img, c, update_emas=update_emas, cam_pred=cam_pred)
        return {'logits': results['score'], 'cam': results['cam'], 'nocond_output': results['nocond_output']}

    def run_IR(self, input_img, sync=None):
        if input_img.size(2) != 64:
            scale_factor = 64 / input_img.size(2)
            input_img = F.interpolate(input_img, scale_factor=scale_factor, mode='bilinear')
        results = self.IR(input_img)
        return results
    
    def normal_loss_calc(self, img, sync):
        # Only in GeoD mode
        input_img = img['image']
        if input_img.size(2) != 64:
            input_img = F.interpolate(input_img.clone(), scale_factor=64 / input_img.size(2), mode='bilinear')
        generated_normal = img['normal_map']
        generated_mask = ((img['normal_map'][:,0]==0) & (img['normal_map'][:,1]==0) & (img['normal_map'][:,2]==0)).float().unsqueeze(1)
        generated_mask = 1 - generated_mask
        results = self.run_IR(input_img)
        inverse_normal = results['recon_normal'][:input_img.shape[0]].permute(0,3,1,2)
        recon_mask = results['recon_im_mask_both'][:input_img.shape[0]]
        # if generated_normal.shape != inverse_normal.shape or generated_normal.shape[0] != input_img.shape[0]:
        #     print(generated_normal.shape, input_img.shape, recon_mask.shape, inverse_normal.shape)
        normal_loss = ((generated_normal-inverse_normal).abs() * recon_mask * generated_mask).sum() / (recon_mask * generated_mask).sum()
        return normal_loss
    
    
    def photometric_loss(self, im1, im2, mask=None, conf_sigma=None):
        loss = (im1-im2).abs()
        if conf_sigma is not None:
            loss = loss *2**0.5 / (conf_sigma +EPS) + (conf_sigma +EPS).log()
        if mask is not None:
            mask = mask.expand_as(loss)
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()
        return loss

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg, batch_idx=None):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth', 'IRboth']
        if self.G.rendering_kwargs.get('density_reg', 0) == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        r1_gamma = self.r1_gamma

        alpha = min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3), 1) if self.gpc_reg_fade_kimg > 0 else 1
        swapping_prob = (1 - alpha) * 1 + alpha * self.gpc_reg_prob if self.gpc_reg_prob is not None else None

        if self.neural_rendering_resolution_final is not None:
            alpha = min(cur_nimg / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (1 - alpha) + self.neural_rendering_resolution_final * alpha))
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial

        real_img_raw = filtered_resizing(real_img, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)

        if self.blur_raw_target:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(blur_sigma).square().neg().exp2()
                real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())

        real_img = {'image': real_img, 'image_raw': real_img_raw}

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                G_outputs = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, 
                                flip=False, flip_type=self.G.rendering_kwargs.get('flip_type', 'flip'))
                gen_img, cam2world = G_outputs['gen_output'], G_outputs['cam']
                if self.dis_cam_weight > 0:
                    d_cond = self.run_D(gen_img, None, blur_sigma=blur_sigma, cam_pred=True)['cam']
                    d_results = self.run_D(gen_img, d_cond.detach()[:,:2], blur_sigma=blur_sigma)
                    
                else:
                    d_results = self.run_D(gen_img, None, blur_sigma=blur_sigma)
                gen_logits = d_results['logits']
                if d_results['nocond_output'] is not None:
                    nocond_output = d_results['nocond_output']
                    training_stats.report('Loss/scores/fake_nocond', nocond_output)
                    training_stats.report('Loss/signs/fake_nocond', nocond_output.sign())
                    loss_Gmain = torch.nn.functional.softplus(-nocond_output)
                else:
                    loss_Gmain = 0

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = loss_Gmain + torch.nn.functional.softplus(-gen_logits)
                training_stats.report('Loss/G/loss', loss_Gmain)


            with torch.autograd.profiler.record_function('Gmain_backward'):
                    (loss_Gmain).mean().mul(gain).backward()

            if self.G.rendering_kwargs.get('flip_to_dis', False):
                
                with torch.autograd.profiler.record_function('Gmain_flip_forward'):
                    G_outputs_flip = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, 
                                    flip=False, flip_type=self.G.rendering_kwargs.get('flip_type', 'flip'), cam_flip=True, detach_pose=self.detach_pose)
                    gen_img_flip = G_outputs_flip['gen_output']
                    if self.dis_cam_weight > 0:
                        d_cond_flip = self.run_D(gen_img_flip, None, blur_sigma=blur_sigma, cam_pred=True)['cam']
                        d_results_flip = self.run_D(gen_img_flip, d_cond_flip.detach()[:,:2], blur_sigma=blur_sigma)
                    else:
                        d_results_flip = self.run_D(gen_img_flip, None, blur_sigma=blur_sigma)
                    gen_logits_flip = d_results_flip['logits']

                    if d_results_flip['nocond_output'] is not None:
                        nocond_output_flip = d_results_flip['nocond_output']
                        training_stats.report('Loss/scores/fake_flip_nocond', nocond_output_flip)
                        training_stats.report('Loss/signs/fake_flip_nocond', nocond_output_flip.sign())
                        loss_Gmain_flip = torch.nn.functional.softplus(-nocond_output_flip)
                    else:
                        loss_Gmain_flip = 0

                    training_stats.report('Loss/scores/fake_flip', gen_logits_flip)
                    training_stats.report('Loss/signs/fake_flip', gen_logits_flip.sign())
                    loss_Gmain_flip = loss_Gmain_flip + torch.nn.functional.softplus(-gen_logits_flip)
                    training_stats.report('Loss/G/loss_flip', loss_Gmain_flip)

                with torch.autograd.profiler.record_function('Gmain_flip_backward'):
                    loss_Gmain_flip.mean().mul(gain).backward()
            
            #GeoD
            if self.geod and cur_nimg > self.normal_start_step * 1000 and cur_nimg < self.normal_gamma_stop * 1000:
                assert batch_idx is not None
                flag = True
                if cur_nimg > self.geod_interval_start * 1000:
                    if batch_idx % self.geod_interval != 0:
                        flag = False
                if flag:
                    with torch.autograd.profiler.record_function('normal_forward'):
                        G_outputs = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, 
                                        flip=False, flip_type=self.G.rendering_kwargs.get('flip_type', 'flip'))
                        gen_img = G_outputs['gen_output']
                        if cur_nimg > self.normal_start_step * 1000 and cur_nimg < self.normal_gamma_stop * 1000:
                            if self.normal_end_step == -1:
                                normal_gamma = self.normal_gamma_start
                            else:
                                temp = self.normal_gamma_start + (self.normal_gamma_end-self.normal_gamma_start)*(cur_nimg/1000-self.normal_start_step)/(self.normal_end_step-self.normal_start_step+1e-7)
                                normal_gamma = min(temp, self.normal_gamma_end)
                        else:
                            normal_gamma = 0

                        loss_normal = self.normal_loss_calc(gen_img, sync=True) * normal_gamma

                        training_stats.report('Loss/G/normal_loss', loss_normal)

                    with torch.autograd.profiler.record_function('normal_backward'):
                        loss_normal.mean().mul(gain).backward()
            
            # Experimentally found this regularization can help align the canonical view.
            if self.pose_gd_weight > 0:
                with torch.autograd.profiler.record_function('Posegloss_forward'):
                    mapping_results = self.G.mapping(gen_z, gen_c, update_emas=False, detach_w=True)
                    cam_pose = mapping_results['cam_pose']
                    if self.learn_fov:
                        gen_output = self.G.synthesis(mapping_results['ws'], gen_c, cam2world_matrix=mapping_results['c2w'], neural_rendering_resolution=neural_rendering_resolution, update_emas=False, flip=False, flip_type=self.G.rendering_kwargs.get('flip_type', 'flip'), fov=cam_pose[:,2:3])
                    else:
                        gen_output = self.G.synthesis(mapping_results['ws'], gen_c, cam2world_matrix=mapping_results['c2w'], neural_rendering_resolution=neural_rendering_resolution, update_emas=False, flip=False, flip_type=self.G.rendering_kwargs.get('flip_type', 'flip'))
                    pose_d = self.run_D(gen_output, None, blur_sigma=blur_sigma, cam_pred=True)['cam']
                    loss_poseg = F.mse_loss(cam_pose[:,:2], pose_d.detach()) * self.pose_gd_weight
                    training_stats.report('Loss/G/posegloss', loss_poseg)
                with torch.autograd.profiler.record_function('Posegloss_backward'):
                    loss_poseg.mul(gain).backward()


        # Density Regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'l1':

            mapping_results = self.G.mapping(gen_z, gen_c, update_emas=False)
            ws = mapping_results['ws']
            cam2world = mapping_results['c2w']
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)['ws'][:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * self.G.rendering_kwargs['density_reg_p_dist']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Alternative density regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-detach':

            ws = self.G.mapping(gen_z, gen_c, update_emas=False)['ws']

            initial_coordinates = torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1 # Front

            perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws.device) * (1/256) * self.G.rendering_kwargs['box_warp'] # Behind
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            monotonic_loss = torch.relu(sigma_initial.detach() - sigma_perturbed).mean() * 10
            monotonic_loss.mul(gain).backward()


            ws = self.G.mapping(gen_z, gen_c, update_emas=False)['ws']
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)['ws'][:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * (1/256) * self.G.rendering_kwargs['box_warp']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Alternative density regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-fixed':

            ws = self.G.mapping(gen_z, gen_c, update_emas=False)['ws']

            initial_coordinates = torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1 # Front

            perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws.device) * (1/256) * self.G.rendering_kwargs['box_warp'] # Behind
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            monotonic_loss = torch.relu(sigma_initial - sigma_perturbed).mean() * 10
            monotonic_loss.mul(gain).backward()


            ws = self.G.mapping(gen_z, gen_c, update_emas=False)['ws']
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)['ws'][:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * (1/256) * self.G.rendering_kwargs['box_warp']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()


            

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if self.G.rendering_kwargs.get('flip_to_disd', False):
            loss_dgen_weight = self.G.rendering_kwargs.get('flip_to_disd_weight', 0.5)
        else:
            loss_dgen_weight = 1.0
        if phase in ['Dmain', 'Dboth']:
            # Update camera head
            if self.dis_cam_weight > 0: # can change the pose to the flipped view?
                with torch.autograd.profiler.record_function('Dcam_forward'):
                    if self.rnd_pose_ford:
                        G_outputs = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, update_emas=True,
                                                flip=False, flip_type=self.G.rendering_kwargs.get('flip_type', 'flip'), cam_flip=True)
                    else:
                        G_outputs = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, update_emas=True)
                    gen_img = G_outputs['gen_output']
                    cam_pose = G_outputs['cam_pose'].detach()
                    cam_pose_pred = self.run_D(gen_img, None, blur_sigma=blur_sigma, update_emas=True, cam_pred=True)['cam']
                    loss_Dcam = torch.nn.functional.mse_loss(cam_pose[:,:2], cam_pose_pred)
                    training_stats.report('Loss/D/loss_Dcam', loss_Dcam)
                with torch.autograd.profiler.record_function('Dcam_backward'):
                    (loss_Dcam * self.dis_cam_weight).mean().mul(gain).backward()
            
            if self.pose_gd_weight > 0 and self.dis_cam_weight == 0: # will only use the pose from the generator to train D
                with torch.autograd.profiler.record_function('Posegloss_forward'):
                    mapping_results = self.G.mapping(gen_z, gen_c, update_emas=False, detach_w=True)
                    cam_pose = mapping_results['cam_pose']
                    if self.learn_fov:
                        gen_output = self.G.synthesis(mapping_results['ws'], gen_c, cam2world_matrix=mapping_results['c2w'], neural_rendering_resolution=neural_rendering_resolution, update_emas=False, flip=False, flip_type=self.G.rendering_kwargs.get('flip_type', 'flip'), fov=cam_pose[:,2:3])
                    else:
                        gen_output = self.G.synthesis(mapping_results['ws'], gen_c, cam2world_matrix=mapping_results['c2w'], neural_rendering_resolution=neural_rendering_resolution, update_emas=False, flip=False, flip_type=self.G.rendering_kwargs.get('flip_type', 'flip'))
                    pose_d = self.run_D(gen_output, None, blur_sigma=blur_sigma, cam_pred=True)['cam']
                    loss_poseg = F.mse_loss(cam_pose.detach()[:,:2], pose_d) * self.pose_gd_weight
                    training_stats.report('Loss/G/posegloss', loss_poseg)
                with torch.autograd.profiler.record_function('Posegloss_backward'):
                    loss_poseg.mul(gain).backward()


            with torch.autograd.profiler.record_function('Dgen_forward'):
                G_outputs = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, update_emas=True)
                gen_img = G_outputs['gen_output']
                if self.dis_cam_weight > 0:
                    d_cond = self.run_D(gen_img, None, blur_sigma=blur_sigma, update_emas=True, cam_pred=True)['cam']
                    d_results = self.run_D(gen_img, d_cond.detach()[:,:2], blur_sigma=blur_sigma, update_emas=True)
                    
                else:
                    d_results = self.run_D(gen_img, None, blur_sigma=blur_sigma, update_emas=True)
                
                if d_results['nocond_output'] is not None:
                    nocond_output = d_results['nocond_output']
                    training_stats.report('Loss/scores/fake_nocond', nocond_output)
                    training_stats.report('Loss/signs/fake_nocond', nocond_output.sign())
                    loss_Dgen = torch.nn.functional.softplus(-nocond_output)
                else:
                    loss_Dgen = 0

                gen_logits = d_results['logits']
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = loss_Dgen + torch.nn.functional.softplus(gen_logits)
                training_stats.report('Loss/D/loss_Dgen', loss_Dgen)
            with torch.autograd.profiler.record_function('Dgen_backward'):
                (loss_Dgen * loss_dgen_weight).mean().mul(gain).backward()

            if self.G.rendering_kwargs.get('flip_to_disd', False):
                with torch.autograd.profiler.record_function('Dgen_flip_forward'):
                    G_outputs_flip = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, update_emas=True,
                                                flip=False, flip_type=self.G.rendering_kwargs.get('flip_type', 'flip'), cam_flip=True)
                    gen_img_flip = G_outputs_flip['gen_output']
                    if self.dis_cam_weight > 0:
                        d_cond_flip = self.run_D(gen_img_flip, None, blur_sigma=blur_sigma, update_emas=True, cam_pred=True)['cam']
                        d_results_flip = self.run_D(gen_img_flip, d_cond_flip.detach()[:,:2], blur_sigma=blur_sigma, update_emas=True)
                    else:
                        d_results_flip = self.run_D(gen_img_flip, None, blur_sigma=blur_sigma, update_emas=True)

                    if d_results_flip['nocond_output'] is not None:
                        nocond_output_flip = d_results_flip['nocond_output']
                        training_stats.report('Loss/scores/fake_flip_nocond', nocond_output_flip)
                        training_stats.report('Loss/signs/fake_flip_nocond', nocond_output_flip.sign())
                        loss_Dgen_flip = torch.nn.functional.softplus(-nocond_output_flip)
                    else:
                        loss_Dgen_flip = 0
                    
                    gen_logits_flip = d_results_flip['logits']
                    training_stats.report('Loss/scores/fake_flip', gen_logits_flip)
                    training_stats.report('Loss/signs/fake_flip', gen_logits_flip.sign())
                    loss_Dgen_flip = loss_Dgen_flip + torch.nn.functional.softplus(gen_logits_flip)
                    training_stats.report('Loss/D/loss_Dgen_flip', loss_Dgen_flip)
                with torch.autograd.profiler.record_function('Dgen_flip_backward'):
                    (loss_Dgen_flip * loss_dgen_weight).mean().mul(gain).backward()
            

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp_image = real_img['image'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp_image_raw = real_img['image_raw'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw}
                if self.dis_cam_weight > 0:
                    d_cond_real = self.run_D(real_img_tmp, None, blur_sigma=blur_sigma, cam_pred=True)['cam']
                    d_results = self.run_D(real_img_tmp, d_cond_real.detach()[:,:2], blur_sigma=blur_sigma)
                else:
                    d_results = self.run_D(real_img_tmp, None, blur_sigma=blur_sigma)

                if d_results['nocond_output'] is not None:
                    real_logits_nocond = d_results['nocond_output']
                    training_stats.report('Loss/scores/real_nocond', real_logits_nocond)
                    training_stats.report('Loss/signs/real_nocond', real_logits_nocond.sign())
                else:
                    real_logits_nocond = None
                
                real_logits = d_results['logits']
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    if real_logits_nocond is not None:
                        loss_Dreal = loss_Dreal + torch.nn.functional.softplus(-real_logits_nocond)
                    if self.G.rendering_kwargs.get('flip_to_disd', False):
                        training_stats.report('Loss/D/loss', loss_Dgen*loss_dgen_weight + loss_Dreal + loss_Dgen_flip*loss_dgen_weight)
                    else:
                        training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    if self.dual_discrimination:
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            if real_logits_nocond is not None:
                                r1_grads = torch.autograd.grad(outputs=[real_logits.sum()+real_logits_nocond.sum()], inputs=[real_img_tmp['image'], real_img_tmp['image_raw']], create_graph=True, only_inputs=True)
                            else:
                                r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image'], real_img_tmp['image_raw']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                            r1_grads_image_raw = r1_grads[1]
                        r1_penalty = r1_grads_image.square().sum([1,2,3]) + r1_grads_image_raw.square().sum([1,2,3])
                        loss_Dr1 = r1_penalty * (r1_gamma / 2)
                    else: # single discrimination
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            if real_logits_nocond is not None:
                                r1_grads = torch.autograd.grad(outputs=[real_logits.sum()+real_logits_nocond.sum()], inputs=[real_img_tmp['image']], create_graph=True, only_inputs=True)
                            else:
                                r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                        r1_penalty = r1_grads_image.square().sum([1,2,3])
                        loss_Dr1 = r1_penalty * r1_gamma
                    
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

        # GeoD
        if phase in ['IRboth']:
            with torch.autograd.profiler.record_function('IR_forward'):
                if isinstance(real_img, dict):
                    real_img = real_img['image']
                if real_img.size(2) != 64:
                    scale_factor = 64 / real_img.size(2)
                    real_img = F.interpolate(real_img, scale_factor=scale_factor, mode='bilinear')
                real_img64 = real_img.detach()
                real_img64.requires_grad = True
                b = real_img64.shape[0]
                IR_results = self.run_IR(real_img64, sync=True)
                loss_l1_im = self.photometric_loss(IR_results['recon_im'][:b], real_img64, mask=IR_results['recon_im_mask_both'][:b], conf_sigma=IR_results['conf_sigma_l1'])
                loss_l1_im_flip = self.photometric_loss(IR_results['recon_im'][b:], real_img64, mask=IR_results['recon_im_mask_both'][b:], conf_sigma=IR_results['conf_sigma_l1_flip'])
                loss_perc_im = self.PerceptualLoss(IR_results['recon_im'][:b], real_img64, mask=IR_results['recon_im_mask_both'][:b], conf_sigma=IR_results['conf_sigma_percl'])
                loss_perc_im_flip = self.PerceptualLoss(IR_results['recon_im'][b:], real_img64, mask=IR_results['recon_im_mask_both'][b:], conf_sigma=IR_results['conf_sigma_percl_flip'])
                loss_depth_sm = ((IR_results['canon_depth'][:,:-1,:] - IR_results['canon_depth'][:,1:,:]) /(self.max_depth-self.min_depth)).abs().mean()
                loss_depth_sm += ((IR_results['canon_depth'][:,:,:-1] - IR_results['canon_depth'][:,:,1:]) /(self.max_depth-self.min_depth)).abs().mean()
                lam_flip = self.lam_flip
                loss_IR = loss_l1_im + lam_flip*loss_l1_im_flip + self.lam_perc*(loss_perc_im + lam_flip*loss_perc_im_flip) + self.lam_depth_sm*loss_depth_sm
                training_stats.report('Loss/IR/ir_loss', loss_IR)
            
            with torch.autograd.profiler.record_function('IR_backward'):
                loss_IR.mean().backward()


class PerceptualLoss(nn.Module):
    def __init__(self, requires_grad=False):
        super(PerceptualLoss, self).__init__()
        mean_rgb = torch.FloatTensor([0.485, 0.456, 0.406])
        std_rgb = torch.FloatTensor([0.229, 0.224, 0.225])
        self.register_buffer('mean_rgb', mean_rgb)
        self.register_buffer('std_rgb', std_rgb)

        vgg_pretrained_features = torchvision.models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def normalize(self, x):
        out = x/2 + 0.5
        out = (out - self.mean_rgb.view(1,3,1,1)) / self.std_rgb.view(1,3,1,1)
        return out

    def __call__(self, im1, im2, mask=None, conf_sigma=None):
        im = torch.cat([im1,im2], 0)
        im = self.normalize(im)  # normalize input

        ## compute features
        feats = []
        f = self.slice1(im)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice2(f)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice3(f)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice4(f)
        feats += [torch.chunk(f, 2, dim=0)]

        losses = []
        for f1, f2 in feats[2:3]:  # use relu3_3 features only
            loss = (f1-f2)**2
            if conf_sigma is not None:
                loss = loss / (2*conf_sigma**2 +EPS) + (conf_sigma +EPS).log()
            if mask is not None:
                b, c, h, w = loss.shape
                _, _, hm, wm = mask.shape
                sh, sw = hm//h, wm//w
                mask0 = nn.functional.avg_pool2d(mask, kernel_size=(sh,sw), stride=(sh,sw)).expand_as(loss)
                loss = (loss * mask0).sum() / mask0.sum()
            else:
                loss = loss.mean()
            losses += [loss]
        return sum(losses)



#----------------------------------------------------------------------------
