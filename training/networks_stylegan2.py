# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Network architectures from the paper
"Analyzing and Improving the Image Quality of StyleGAN".
Matches the original implementation of configs E-F by Karras et al. at
https://github.com/NVlabs/stylegan2/blob/master/training/networks_stylegan2.py"""

import numpy as np
import torch
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma
import math
import torch.nn as nn
from einops import repeat, rearrange
import torch.nn.functional as F

#----------------------------------------------------------------------------

@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

#----------------------------------------------------------------------------

@misc.profiled_function
def modulated_conv2d(
    x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,                     # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,                     # Modulation coefficients of shape [batch_size, in_channels].
    noise           = None,     # Optional noise tensor to add to the output activations.
    up              = 1,        # Integer upsampling factor.
    down            = 1,        # Integer downsampling factor.
    padding         = 0,        # Padding with respect to the upsampled image.
    resample_filter = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate      = True,     # Apply weight demodulation?
    flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings(): # this value will be treated as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class Conv2dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        kernel_size,                    # Width and height of the convolution kernel.
        bias            = True,         # Apply additive bias before the activation function?
        activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
        up              = 1,            # Integer upsampling factor.
        down            = 1,            # Integer downsampling factor.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
        channels_last   = False,        # Expect the input to have memory_format=channels_last?
        trainable       = True,         # Update the weights of this layer during training?
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1) # slightly faster
        x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

    def extra_repr(self):
        return ' '.join([
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, activation={self.activation:s},',
            f'up={self.up}, down={self.down}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.998,    # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if update_emas and self.w_avg_beta is not None:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

    def extra_repr(self):
        return f'z_dim={self.z_dim:d}, c_dim={self.c_dim:d}, w_dim={self.w_dim:d}, num_ws={self.num_ws:d}'


# Mapping Network in PoF3D
@persistence.persistent_class
class MappingNetworkPose(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier_pose   = 0.01,
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.998,    # Decay for tracking the moving average of W during training, None = do not track.
        linear_pose     = False,
        cam_dim         = 2,
        learn_fov       = False,
        uniform_sampling= False

    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta
        self.linear_pose = linear_pose
        self.cam_dim = cam_dim
        self.learn_fov = learn_fov
        self.uniform_sampling = uniform_sampling

        if embed_features is None:
            embed_features = w_dim
        # if c_dim == 0:
        embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        # if c_dim > 0:
        #     self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))
        
        if self.linear_pose:
            self.affine_layer = FullyConnectedLayer(w_dim, self.cam_dim, activation='linear', lr_multiplier=lr_multiplier_pose)
        else:
            self.affine_layer1 = FullyConnectedLayer(w_dim, w_dim, activation='lrelu', lr_multiplier=lr_multiplier_pose)
            self.affine_layer2 = FullyConnectedLayer(w_dim, self.cam_dim, activation='linear', lr_multiplier=lr_multiplier_pose)



    def forward(self, z=None, c=None, truncation_psi=1, truncation_cutoff=None, update_emas=False, detach_w=False, clamp_h=None, clamp_v=None, h_mean=math.pi/2, r_mean=2.7, fov_mean=20):
        # Embed, normalize, and concat inputs.
        x = None
        cam_params_avg = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))
            # if self.c_dim > 0:
            #     misc.assert_shape(c, [None, self.c_dim])
            #     y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
            #     x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)
        
        before_repeat_w = x.clone()
        if detach_w:
            cam_temp = self.affine_layer1(x.detach())
            cam_params = self.affine_layer2(cam_temp)
            cam_params[:, 0:1] += h_mean
            cam_params[:, 1:2] += math.pi/2
            if self.cam_dim == 3:
                if self.learn_fov:
                    cam_params[:, 2:3] += fov_mean
                else:
                    cam_params[:, 2:3] += r_mean
        else:
            #clamp version
            if self.linear_pose:
                cam_params = self.affine_layer(x) + math.pi/2

            else:
                cam_temp = self.affine_layer1(x)
                cam_params = self.affine_layer2(cam_temp)
                cam_params[:, 0:1] += h_mean
                cam_params[:, 1:2] += math.pi/2
                if self.cam_dim == 3:
                    if self.learn_fov:
                        cam_params[:, 2:3] += fov_mean
                    else:
                        cam_params[:, 2:3] += r_mean
        
        if self.uniform_sampling:
            cam_params[:, 0:1] = cam_params[:, 0:1] * 0 + torch.rand((cam_params.shape[0], 1), device=cam_params.device) * math.pi * 2
            cam_params[:, 1:2] = cam_params[:, 1:2] * 0 + (torch.rand((cam_params.shape[0], 1), device=cam_params.device)*2-1) * math.pi/8 + math.pi/2
        
        if clamp_h is not None:
            cam_params[:, 0:1] = torch.clamp(cam_params[:, 0:1].clone(), clamp_h[0], clamp_h[1])
        if clamp_v is not None:
            cam_params[:, 1:2] = torch.clamp(cam_params[:, 1:2].clone(), clamp_v[0], clamp_v[1])
            cam_params[:, 1:2] = torch.clamp(cam_params[:, 1:2].clone(), 1e-5, math.pi-(1e-5))

        # Update moving average of W.
        if update_emas and self.w_avg_beta is not None:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        
        return {'ws':x, 
                'before_repeat_w': before_repeat_w,
                'cam_params': cam_params, 
                'cam_params_avg': cam_params_avg}

    def extra_repr(self):
        return f'z_dim={self.z_dim:d}, c_dim={self.c_dim:d}, w_dim={self.w_dim:d}, num_ws={self.num_ws:d}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        w_dim,                          # Intermediate latent (W) dimensionality.
        resolution,                     # Resolution of this layer.
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        use_noise       = True,         # Enable noise input?
        activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last   = False,        # Use channels_last format for the weights?
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        assert noise_mode in ['random', 'const', 'none']
        in_resolution = self.resolution // self.up
        misc.assert_shape(x, [None, self.in_channels, in_resolution, in_resolution])
        styles = self.affine(w)

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1) # slightly faster
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up,
            padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

    def extra_repr(self):
        return ' '.join([
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d},',
            f'resolution={self.resolution:d}, up={self.up}, activation={self.activation:s}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class ToRGBLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None, channels_last=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain
        if x.size(0) > styles.size(0):
            assert (x.size(0) // styles.size(0) * styles.size(0) == x.size(0))
            styles = repeat(styles, 'b c -> (b s) c', s=x.size(0) // styles.size(0))
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x

    def extra_repr(self):
        return f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                            # Number of input channels, 0 = first block.
        out_channels,                           # Number of output channels.
        w_dim,                                  # Intermediate latent (W) dimensionality.
        resolution,                             # Resolution of this block.
        img_channels,                           # Number of output color channels.
        is_last,                                # Is this the last block?
        architecture            = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter         = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp              = 256,          # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16                = False,        # Use FP16 for this block?
        fp16_channels_last      = False,        # Use channels-last memory format with FP16?
        fused_modconv_default   = True,         # Default value of fused_modconv. 'inference_only' = True for inference, False for training.
        **layer_kwargs,                         # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.fused_modconv_default = fused_modconv_default
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution]))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2,
                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
            conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=2,
                resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, ws, force_fp32=False, fused_modconv=None, update_emas=False, **layer_kwargs):
        _ = update_emas # unused
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        if ws.device.type != 'cuda':
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            fused_modconv = self.fused_modconv_default
        if fused_modconv == 'inference_only':
            fused_modconv = (not self.training)

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        else:
            misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

        # ToRGB.
        if img is not None:
            misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            img = upfirdn2d.upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 4,        # Use FP16 for the N highest resolutions.
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.num_fp16_res = num_fp16_res
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img

    def extra_repr(self):
        return ' '.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'num_fp16_res={self.num_fp16_res:d}'])



@persistence.persistent_class
class Style2Layer(nn.Module):
    def __init__(self, 
        in_channels, 
        out_channels, 
        w_dim, 
        activation='lrelu', 
        resample_filter=[1,3,3,1],
        magnitude_ema_beta = -1,           # -1 means not using magnitude ema
        **unused_kwargs):

        # simplified version of SynthesisLayer 
        # no noise, kernel size forced to be 1x1, used in NeRF block
        super().__init__()
        self.activation = activation
        self.conv_clamp = None
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = 0
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        self.w_dim = w_dim
        self.in_features = in_channels
        self.out_features = out_channels
        memory_format = torch.contiguous_format

        if w_dim > 0:
            self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
            self.weight = torch.nn.Parameter(
               torch.randn([out_channels, in_channels, 1, 1]).to(memory_format=memory_format))
            self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        
        else:
            self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels))
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
            self.weight_gain = 1.

            # initialization
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        self.magnitude_ema_beta = magnitude_ema_beta
        if magnitude_ema_beta > 0:
            self.register_buffer('w_avg', torch.ones([]))

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, style={}'.format(
            self.in_features, self.out_features, self.w_dim
        )

    def forward(self, x, w=None, fused_modconv=None, gain=1, up=1, **unused_kwargs):
        flip_weight = True # (up == 1) # slightly faster HACK
        act = self.activation

        if (self.magnitude_ema_beta > 0):
            if self.training:  # updating EMA.
                with torch.autograd.profiler.record_function('update_magnitude_ema'):
                    magnitude_cur = x.detach().to(torch.float32).square().mean()
                    self.w_avg.copy_(magnitude_cur.lerp(self.w_avg, self.magnitude_ema_beta))
            input_gain = self.w_avg.rsqrt()
            x = x * input_gain

        if fused_modconv is None:
            with misc.suppress_tracer_warnings(): # this value will be treated as a constant
                fused_modconv = not self.training
        
        if self.w_dim > 0:           # modulated convolution
            assert x.ndim == 4,  "currently not support modulated MLP"
            styles = self.affine(w)      # Batch x style_dim
            if x.size(0) > styles.size(0):
                styles = repeat(styles, 'b c -> (b s) c', s=x.size(0) // styles.size(0))
            
            x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=None, up=up,
                padding=self.padding, resample_filter=self.resample_filter, 
                flip_weight=flip_weight, fused_modconv=fused_modconv)
            act_gain = self.act_gain * gain
            act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
            x = bias_act.bias_act(x, self.bias.to(x.dtype), act=act, gain=act_gain, clamp=act_clamp)
        
        else:
            if x.ndim == 2:  # MLP mode
                x = F.relu(F.linear(x, self.weight, self.bias.to(x.dtype)))
            else:
                x = F.relu(F.conv2d(x, self.weight[:,:,None, None], self.bias))
                # x = bias_act.bias_act(x, self.bias.to(x.dtype), act='relu')
        return x


@persistence.persistent_class
class NeRFBlock(nn.Module):
    ''' 
    Predicts volume density and color from 3D location, viewing
    direction, and latent code z.
    '''
    # dimensions
    input_dim            = 3
    w_dim                = 512   # style latent
    z_dim                = 0     # input latent
    rgb_out_dim          = 29
    hidden_size          = 128
    n_blocks             = 4
    img_channels         = 3
    magnitude_ema_beta   = -1
    disable_latents      = False
    max_batch_size       = 2 ** 18
    shuffle_factor       = 1
    implementation       = 'batch_reshape'  # option: [flatten_2d, batch_reshape]

    # architecture settings
    activation           = 'lrelu'
    use_skip             = False 
    use_viewdirs         = False
    add_rgb              = False
    predict_rgb          = False
    inverse_sphere       = False
    merge_sigma_feat     = False   # use one MLP for sigma and features
    no_sigma             = False   # do not predict sigma, only output features
    
    tcnn_backend         = False
    use_style            = None 
    use_normal           = False
    use_sdf              = None
    volsdf_exp_beta      = False
    normalized_feat      = False
    final_sigmoid_act    = False

    # positional encoding inpuut
    use_pos              = False
    n_freq_posenc        = 10
    n_freq_posenc_views  = 4
    downscale_p_by       = 1
    gauss_dim_pos        = 20 
    gauss_dim_view       = 4 
    gauss_std            = 10.
    positional_encoding  = "normal"

    def __init__(self, nerf_kwargs):
        super().__init__()
        for key in nerf_kwargs:
            if hasattr(self, key):
                setattr(self, key, nerf_kwargs[key])

        self.sdf_mode = self.use_sdf
        self.use_sdf  = self.use_sdf is not None
        if self.use_sdf == 'volsdf':
            self.density_transform = SDFDensityLaplace(
                params_init={'beta': 0.1}, 
                beta_min=0.0001, 
                exp_beta=self.volsdf_exp_beta)

        # ----------- input module -------------------------
        D = self.input_dim if not self.inverse_sphere else self.input_dim + 1
        if self.positional_encoding == 'gauss':
            rng = np.random.RandomState(2021)
            B_pos  = self.gauss_std * torch.from_numpy(rng.randn(D, self.gauss_dim_pos * D)).float()
            B_view = self.gauss_std * torch.from_numpy(rng.randn(3, self.gauss_dim_view * 3)).float()
            self.register_buffer("B_pos", B_pos)
            self.register_buffer("B_view", B_view)
            dim_embed = D * self.gauss_dim_pos * 2
            dim_embed_view = 3 * self.gauss_dim_view * 2
        elif self.positional_encoding == 'normal':
            dim_embed = D * self.n_freq_posenc * 2
            dim_embed_view = 3 * self.n_freq_posenc_views * 2
        else:  # not using positional encoding
            dim_embed, dim_embed_view = D, 3

        if self.use_pos:
            dim_embed, dim_embed_view = dim_embed + D, dim_embed_view + 3

        self.dim_embed = dim_embed
        self.dim_embed_view = dim_embed_view

        # ------------ Layers --------------------------
        assert not (self.add_rgb and self.predict_rgb), "only one could be achieved"
        assert not ((self.use_viewdirs or self.use_normal) and (self.merge_sigma_feat or self.no_sigma)), \
            "merged MLP does not support."
        
        if self.disable_latents:
            w_dim = 0
        elif self.z_dim > 0:  # if input global latents, disable using style vectors
            w_dim, dim_embed, dim_embed_view = 0, dim_embed + self.z_dim, dim_embed_view + self.z_dim
        else:
            w_dim = self.w_dim

        final_in_dim = self.hidden_size
        if self.use_normal:
            final_in_dim += D

        final_out_dim = self.rgb_out_dim * self.shuffle_factor
        if self.merge_sigma_feat:
            final_out_dim += self.shuffle_factor  # predicting sigma
        if self.add_rgb:
            final_out_dim += self.img_channels

        # start building the model
        
        self.fc_in  = Style2Layer(dim_embed, self.hidden_size, w_dim, activation=self.activation)
        self.num_ws = 1
        self.skip_layer = self.n_blocks // 2 - 1 if self.use_skip else None      
        if self.n_blocks > 1:
            self.blocks = nn.ModuleList([
                Style2Layer(
                    self.hidden_size if i != self.skip_layer else self.hidden_size + dim_embed, 
                    self.hidden_size, 
                    w_dim, activation=self.activation,
                    magnitude_ema_beta=self.magnitude_ema_beta)
                for i in range(self.n_blocks - 1)])
            self.num_ws += (self.n_blocks - 1)

        if not (self.merge_sigma_feat or self.no_sigma):
            self.sigma_out = ToRGBLayer(self.hidden_size, self.shuffle_factor, w_dim, kernel_size=1)
            self.num_ws += 1
        self.feat_out = ToRGBLayer(final_in_dim, final_out_dim, w_dim, kernel_size=1)
        if (self.z_dim == 0 and (not self.disable_latents)):
            self.num_ws += 1
        else:
            self.num_ws = 0        
        
        if self.use_viewdirs:
            assert self.predict_rgb, "only works when predicting RGB"
            self.from_ray = Conv2dLayer(dim_embed_view, final_out_dim, kernel_size=1, activation='linear')
        
        if self.predict_rgb:   # predict RGB over features
            self.to_rgb = Conv2dLayer(final_out_dim, self.img_channels * self.shuffle_factor, kernel_size=1, activation='linear')
        
    def set_steps(self, steps):
        if hasattr(self, "steps"):
            self.steps.fill_(steps)
        
    def transform_points(self, p, views=False):
        p = p / self.downscale_p_by
        if self.positional_encoding == 'gauss':
            B = self.B_view if views else self.B_pos
            p_transformed = positional_encoding(p, B, 'gauss', self.use_pos)
        elif self.positional_encoding == 'normal':
            L = self.n_freq_posenc_views if views else self.n_freq_posenc
            p_transformed = positional_encoding(p, L, 'normal', self.use_pos)
        else:
            p_transformed = p
        return p_transformed

    def forward(self, p_in, ray_d, z_shape=None, z_app=None, ws=None, shape=None, requires_grad=False, impl=None):
        with torch.set_grad_enabled(self.training or self.use_sdf or requires_grad):
            impl = 'mlp' if self.tcnn_backend else impl
            option, p_in = self.forward_inputs(p_in, shape=shape, impl=impl)
            feat, sigma_raw = self.forward_nerf(option, p_in, ray_d,  ws=ws, z_shape=z_shape, z_app=z_app)
        return feat, sigma_raw

    def forward_inputs(self, p_in, shape=None, impl=None):
        # prepare the inputs
        impl = impl if impl is not None else self.implementation
        if (shape is not None) and (impl == 'batch_reshape'):
            height, width, n_steps = shape[1:]
        elif impl == 'flatten_2d':
            (height, width), n_steps = dividable(p_in.shape[1]), 1
        elif impl == 'mlp':
            height, width, n_steps = 1, 1, p_in.shape[1]
        else:
            raise NotImplementedError("looking for more efficient implementation.")        
        p_in = rearrange(p_in, 'b (h w s) d -> (b s) d h w', h=height, w=width, s=n_steps)
        use_normal = self.use_normal or self.use_sdf
        if use_normal:
            p_in.requires_grad_(True)
        return (height, width, n_steps, use_normal), p_in
    
    def forward_nerf(self, option, p_in, ray_d=None, ws=None, z_shape=None, z_app=None):
        height, width, n_steps, use_normal = option
        
        # forward nerf feature networks
        p = self.transform_points(p_in.permute(0,2,3,1))
        if (self.z_dim > 0) and (not self.disable_latents):
            assert (z_shape is not None) and (ws is None)
            z_shape = repeat(z_shape, 'b c -> (b s) h w c', h=height, w=width, s=n_steps)
            p = torch.cat([p, z_shape], -1)
        p = p.permute(0,3,1,2)    # BS x C x H x W

        if height == width == 1:  # MLP
            p = p.squeeze(-1).squeeze(-1)
            
        net = self.fc_in(p, ws[:, 0] if ws is not None else None)
        if self.n_blocks > 1:
            for idx, layer in enumerate(self.blocks):
                ws_i = ws[:, idx + 1] if ws is not None else None
                if (self.skip_layer is not None) and (idx == self.skip_layer):
                    net = torch.cat([net, p], 1)
                net = layer(net, ws_i, up=1)

        # forward to get the final results
        w_idx = self.n_blocks  # fc_in, self.blocks
                
        feat_inputs = [net]
        if not (self.merge_sigma_feat or self.no_sigma):
            ws_i      = ws[:, w_idx] if ws is not None else None
            sigma_out = self.sigma_out(net, ws_i)
            if use_normal:
                gradients, = grad(
                    outputs=sigma_out, inputs=p_in, 
                    grad_outputs=torch.ones_like(sigma_out, requires_grad=False), 
                    retain_graph=True, create_graph=True, only_inputs=True)
                feat_inputs.append(gradients)
    
        ws_i = ws[:, -1] if ws is not None else None
        net = torch.cat(feat_inputs, 1) if len(feat_inputs) > 1 else net
        feat_out = self.feat_out(net, ws_i)  # this is used for lowres output

        if self.merge_sigma_feat:  # split sigma from the feature
            sigma_out, feat_out = feat_out[:, :self.shuffle_factor], feat_out[:, self.shuffle_factor:]
        elif self.no_sigma:
            sigma_out = None
                
        if self.predict_rgb:
            if self.use_viewdirs and ray_d is not None:
                ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
                ray_d = self.transform_points(ray_d, views=True)
                if self.z_dim > 0:
                    ray_d = torch.cat([ray_d, repeat(z_app, 'b c -> b (h w s) c', h=height, w=width, s=n_steps)], -1)
                ray_d = rearrange(ray_d, 'b (h w s) d -> （b s) d h w', h=height, w=width, s=n_steps)
                feat_ray = self.from_ray(ray_d)
                rgb = self.to_rgb(F.leaky_relu(feat_out + feat_ray))
            else:
                rgb = self.to_rgb(feat_out)

            if self.final_sigmoid_act:
                rgb = torch.sigmoid(rgb)    
            if self.normalized_feat:
                feat_out = feat_out / (1e-7 + feat_out.norm(dim=-1, keepdim=True))
            feat_out = torch.cat([rgb, feat_out], 1)

        # transform back
        if feat_out.ndim == 2:  # mlp mode
            sigma_out = rearrange(sigma_out, '(b s) d -> b s d', s=n_steps) if sigma_out is not None else None
            feat_out  = rearrange(feat_out,  '(b s) d -> b s d', s=n_steps)
        else:
            sigma_out = rearrange(sigma_out, '(b s) d h w -> b (h w s) d', s=n_steps) if sigma_out is not None else None
            feat_out  = rearrange(feat_out,  '(b s) d h w -> b (h w s) d', s=n_steps)
        return feat_out, sigma_out


def positional_encoding(p, size, pe='normal', use_pos=False):
    if pe == 'gauss':
        p_transformed = np.pi * p @ size
        p_transformed = torch.cat(
            [torch.sin(p_transformed), torch.cos(p_transformed)], dim=-1)
    else:
        p_transformed = torch.cat([torch.cat(
            [torch.sin((2 ** i) * np.pi * p),
            torch.cos((2 ** i) * np.pi * p)],
            dim=-1) for i in range(size)], dim=-1)
    if use_pos:
        p_transformed = torch.cat([p_transformed, p], -1)
    return p_transformed
#----------------------------------------------------------------------------

@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        bg_kwargs           = None,
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        if bg_kwargs is not None:
            self.bg_nerf = NeRFBlock(bg_kwargs)
            self.num_ws = self.bg_nerf.num_ws
        else:
            self.bg_nerf = None
            self.num_ws = 0
        self.num_ws += self.synthesis.num_ws
        self.mapping = MappingNetworkPose(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        return img

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        tmp_channels,                       # Number of intermediate channels.
        out_channels,                       # Number of output channels.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of input color channels.
        first_layer_idx,                    # Index of the first layer.
        architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
        activation          = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        freeze_layers       = 0,            # Freeze-D: Number of layers to freeze.
    ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0
        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable
        trainable_iter = trainable_gen()

        if in_channels == 0 or architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, tmp_channels, kernel_size=1, activation=activation,
                trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv0 = Conv2dLayer(tmp_channels, tmp_channels, kernel_size=3, activation=activation,
            trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv1 = Conv2dLayer(tmp_channels, out_channels, kernel_size=3, activation=activation, down=2,
            trainable=next(trainable_iter), resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last)

        if architecture == 'resnet':
            self.skip = Conv2dLayer(tmp_channels, out_channels, kernel_size=1, bias=False, down=2,
                trainable=next(trainable_iter), resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, force_fp32=False):
        if (x if x is not None else img).device.type != 'cuda':
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if x is not None:
            misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(img, self.resample_filter) if self.architecture == 'skip' else None

        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings(): # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
        return x

    def extra_repr(self):
        return f'group_size={self.group_size}, num_channels={self.num_channels:d}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        cmap_dim,                       # Dimensionality of mapped conditioning label, 0 = no label.
        resolution,                     # Resolution of this block.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        dis_cam_weight      = 0,        # whether use a camera head
        pose_gd_weight      = 0,
        dis_linear_pose     = False,
        nocond_output       = False
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture
        self.dis_cam_weight = dis_cam_weight
        self.pose_gd_weight = pose_gd_weight
        self.dis_linear_pose = dis_linear_pose
        self.nocond_output = nocond_output

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)

        if self.dis_cam_weight > 0 or self.pose_gd_weight > 0:
            if self.dis_linear_pose:
                self.cam_out = FullyConnectedLayer(in_channels * (resolution ** 2), 2, activation='linear')
            else:
                self.cam_out1 = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels * (resolution ** 2)//2, activation='lrelu')
                self.cam_out2 = FullyConnectedLayer(in_channels * (resolution ** 2)//2, 2, activation='linear')


    def forward(self, x, img, cmap, force_fp32=False):
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution]) # [NCHW]
        _ = force_fp32 # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)

        cam_ = None
        if self.dis_cam_weight > 0 or self.pose_gd_weight > 0:
            if self.dis_linear_pose:
                cam_ = self.cam_out(x.flatten(1))
            else:
                cam_ = self.cam_out1(x.flatten(1))
                cam_ = self.cam_out2(cam_)

        x = self.fc(x.flatten(1))
        x = self.out(x)

        # Conditioning.
        results = {}
        if self.nocond_output:
            results['nocond_output'] = x.clone()
        else:
            results['nocond_output'] = None

        if self.cmap_dim > 0 and cmap is not None:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))
        results['score'] = x
        results['cam'] = cam_
        assert x.dtype == dtype
        return results

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

# @persistence.persistent_class
# class DiscriminatorCamera(torch.nn.Module):
#     def __init__(self,
#         in_channels,                    # Number of input channels.
#         cmap_dim,                       # Dimensionality of mapped conditioning label, 0 = no label.
#         resolution,                     # Resolution of this block.
#         img_channels,                   # Number of input color channels.
#         architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
#         mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
#         mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
#         activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
#         conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
#     ):
#         assert architecture in ['orig', 'skip', 'resnet']
#         super().__init__()
#         self.in_channels = in_channels
#         self.cmap_dim = cmap_dim
#         self.resolution = resolution
#         self.img_channels = img_channels
#         self.architecture = architecture

#         if architecture == 'skip':
#             self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
#         self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
#         self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp)
#         self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
#         self.out = FullyConnectedLayer(in_channels, 2 if cmap_dim == 0 else cmap_dim)

#     def forward(self, x, img, cmap, force_fp32=False):
#         misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution]) # [NCHW]
#         _ = force_fp32 # unused
#         dtype = torch.float32
#         memory_format = torch.contiguous_format

#         # FromRGB.
#         x = x.to(dtype=dtype, memory_format=memory_format)
#         if self.architecture == 'skip':
#             misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
#             img = img.to(dtype=dtype, memory_format=memory_format)
#             x = x + self.fromrgb(img)

#         # Main layers.
#         if self.mbstd is not None:
#             x = self.mbstd(x)
#         x = self.conv(x)
#         x = self.fc(x.flatten(1))
#         x = self.out(x)

#         # Conditioning.
#         if self.cmap_dim > 0:
#             misc.assert_shape(cmap, [None, self.cmap_dim])
#             x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

#         assert x.dtype == dtype
#         return x

#     def extra_repr(self):
#         return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class Discriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)

    def forward(self, img, c, update_emas=False, **block_kwargs):
        _ = update_emas # unused
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'

#----------------------------------------------------------------------------