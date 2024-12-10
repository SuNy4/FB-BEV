# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import BaseModule, auto_fp16
from mmdet.models import NECKS

import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv

class AggregationBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AggregationBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        out = self.gelu(self.conv1(x))
        out = self.gelu(self.conv2(out))
        out = self.gelu(self.conv3(out))
        return out

class UpsampleLayer(nn.Module):
    def __init__(self, scale_factor, mode, in_channels=None, out_channels=None, convtype=None):
        super(UpsampleLayer, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=True)
        if convtype=='2d':
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        if convtype=='3d':
            self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        if convtype==None:
            self.conv=None

    def forward(self, x):
        out = self.upsample(x)
        if self.conv != None:
            out = self.conv(out)
        return out

@NECKS.register_module()
class BEV2DFCN(nn.Module):
    def __init__(self, flatten_height, height, in_channels, mid_channels, h_level, out_channels=None):
        super(BEV2DFCN, self).__init__()
        self.h_level = h_level
        self.flatten_height = flatten_height
        self.conv0 = nn.Conv2d(in_channels*height, mid_channels, kernel_size=3, stride=1, padding=1) if self.flatten_height else None
        self.conv_1 = nn.Conv1d(mid_channels, mid_channels*h_level[0], kernel_size=1)
        self.conv_2 = nn.Conv1d(mid_channels*h_level[0], mid_channels*h_level[1], kernel_size=1)
        self.conv_3 = nn.Conv1d(mid_channels*h_level[1], mid_channels*h_level[2], kernel_size=1)
        self.conv2 = nn.Conv1d(mid_channels, height, kernel_size=1) if self.flatten_height else None
        self.bn_flat0 = nn.BatchNorm2d(mid_channels) if self.flatten_height else None
        self.bn_flat_1 = nn.BatchNorm1d(mid_channels*h_level[0])
        self.bn_flat_2 = nn.BatchNorm1d(mid_channels*h_level[1])
        self.bn_flat2 = nn.BatchNorm2d(height) if self.flatten_height else None
        self.gelu = nn.GELU()
        self.encoder1 = nn.Conv2d(mid_channels, mid_channels*2, kernel_size=4, stride=2, padding=1)
        self.encoder2 = nn.Conv2d(mid_channels*2, mid_channels*4, kernel_size=4, stride=2, padding=1)
        self.decoder1 = nn.ConvTranspose2d(mid_channels*4, mid_channels*2, kernel_size=4, stride=2, padding=1)
        self.decoder2 = nn.ConvTranspose2d(mid_channels*2, mid_channels, kernel_size=4, stride=2, padding=1)
        # self.decoder1 = UpsampleLayer(2, 'bilinear', in_channels=mid_channels*4, out_channels=mid_channels*2, convtype='2d')
        # self.decoder2 = UpsampleLayer(2, 'bilinear', in_channels=mid_channels*2, out_channels=mid_channels, convtype='2d')

        # self.encoder1 = AggregationBlock(out_channels, out_channels*2)
        # self.encoder2 = AggregationBlock(out_channels*2, out_channels*4)
        
        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(mid_channels*2)
        self.bn2 = nn.BatchNorm2d(mid_channels*4)
        self.bn3 = nn.BatchNorm2d(mid_channels*2)
        self.bn4 = nn.BatchNorm2d(mid_channels)

    def forward(self, x):
        if self.flatten_height:
            x = self.gelu(self.bn_flat0(self.conv0(x)))
        
        # Downsample
        # e1 = self.encoder1(x)
        # e2 = self.encoder2(e1)
        e1 = self.gelu(self.bn1(self.encoder1(x))) #50
        e2 = self.gelu(self.bn2(self.encoder2(e1))) #25

        # Upsample
        d1 = self.gelu(self.bn3(self.decoder1(e2))) #50
        d1 = d1 + e1
        d2 = self.gelu(self.bn4(self.decoder2(d1)))
        out = d2 + x
        bs, _, D, W = out.shape

        out = self.gelu(self.bn_flat_1(self.conv_1(out.flatten(2, 3)))) # bs, C, DW
        out = self.gelu(self.bn_flat_2(self.conv_2(out)))
        out = self.conv_3(out).reshape(bs, -1, D, W, self.h_level[2])
        if self.flatten_height:
            bev_h = self.gelu(self.bn_flat2(self.conv2(out.flatten(2, 3)))).reshape(bs, -1, D, W).permute(0, 2, 3, 1) # bs, H, D, W => bs, D, W, H
            bev_h = bev_h.sigmoid()

            return out, bev_h
        else:
            return out

######## Original FCN2D ################
# @NECKS.register_module()
# class BEV2DFCN(nn.Module):
#     def __init__(self, flatten_height, height, in_channels, out_channels):
#         super(BEV2DFCN, self).__init__()
#         self.flatten_height = flatten_height
#         self.in_channels = in_channels
#         # self.conv0 = nn.Conv2d(self.in_channels*height, self.in_channels, kernel_size=1)
#         self.conv1 = nn.Conv2d(self.in_channels*height, out_channels, kernel_size=1)
#         # self.bn_flat0 = nn.BatchNorm2d(self.in_channels*height/2)
#         self.bn_flat1 = nn.BatchNorm2d(out_channels)
#         self.gelu = nn.GELU()
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.decoder1 = UpsampleLayer(2, 'bilinear', out_channels, out_channels, convtype='2d')
#         self.decoder2 = UpsampleLayer(2, 'bilinear', out_channels, out_channels, convtype='2d')

#         # Batch Normalization
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.bn2 = nn.BatchNorm2d(out_channels)

#     def forward(self, x):
#         if self.flatten_height:
#             # x = self.bn_flat0(self.conv0(x))
#             x = self.bn_flat1(self.conv1(x))
#             x = self.gelu(x)
        
#         # Downsample
#         # e1 = self.encoder1(x)
#         # e2 = self.encoder2(e1)
#         e1 = self.maxpool(x)
#         e2 = self.maxpool(e1)

#         # Upsample
#         d1 = self.gelu(self.bn1(self.decoder1(e2)))
#         d1 = d1 + e1
        
#         d2 = self.gelu(self.bn2(self.decoder2(d1)))
#         out = d2 + x

#         return out

@NECKS.register_module()
class BEV3DFCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BEV3DFCN, self).__init__()
        self.in_channels = in_channels
        self.gelu = nn.GELU()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.decoder1 = UpsampleLayer(2, 'trilinear', out_channels, out_channels, convtype='3d')
        self.decoder2 = UpsampleLayer(2, 'trilinear', out_channels, out_channels, convtype='3d')
        # self.encoder1 = AggregationBlock(out_channels, out_channels*2)
        # self.encoder2 = AggregationBlock(out_channels*2, out_channels*4)
        
        # self.decoder1 = nn.ConvTranspose2d(out_channels*4, out_channels*2, padding=1, kernel_size=4, stride=2)
        # self.decoder2 = nn.ConvTranspose2d(out_channels*2, out_channels, padding=1, kernel_size=4, stride=2)
        
        # Batch Normalization
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):

        # Downsample
        # e1 = self.encoder1(x)
        # e2 = self.encoder2(e1)
        x = self.conv(x)
        e1 = self.maxpool(x)
        e2 = self.maxpool(e1)

        # Upsample
        d1 = self.gelu(self.bn1(self.decoder1(e2)))
        d1 = d1 + e1
        
        d2 = self.gelu(self.bn2(self.decoder2(d1)))
        out = d2 + x

        return out
    
@NECKS.register_module()
class OcclusionMask(nn.Module):
    def __init__(self, in_channel):
        super(OcclusionMask, self).__init__()
        self.in_channel = in_channel
        self.gelu = nn.GELU()
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.bn = nn.BatchNorm3d(in_channel)

    def forward(self, input):
        
        mask = self.bn(input).sum(dim=1)
        mask = self.gelu(mask)
        mask = mask <= 0.7
        mask = torch.any(mask, dim=-1)

        return mask  
    
@NECKS.register_module()
class SparseConv3D(nn.Module):
    def __init__(self, in_channel=None, out_channel = None):
        super(SparseConv3D, self).__init__()
        self.spconv3d = spconv.SparseConv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, input_tensor=None):
        # input: bs, C*H, D, W
        bs, _, D, W = input_tensor
        coords = torch.nonzero(input_tensor, as_tuple=False) # (N, 4)
        sparse_input = input_tensor[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]]
        sparse_input = spconv.SparseConvTensor(
            sparse_input,
            coords = coords[:, [0, 2, 3]],
            spatial_shape = (D, W),
            batch_size = bs
        )
        output = self.spconv2d(sparse_input)
        output = output.dense()

@NECKS.register_module()
class AdaptiveMixing(nn.Module):
    """Adaptive Mixing"""
    def __init__(self, in_dim, in_points, n_groups=1, query_dim=None, out_dim=None, out_points=None):
        super(AdaptiveMixing, self).__init__()

        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim

        self.query_dim = query_dim
        self.in_dim = in_dim
        self.in_points = in_points
        self.n_groups = n_groups
        self.out_dim = out_dim
        self.out_points = out_points

        self.eff_in_dim = in_dim // n_groups
        self.eff_out_dim = out_dim // n_groups

        self.m_parameters = self.eff_in_dim * self.eff_out_dim
        self.s_parameters = self.in_points * self.out_points
        self.total_parameters = self.m_parameters + self.s_parameters

        self.parameter_generator = nn.Linear(self.query_dim, self.n_groups * self.total_parameters)
        self.out_proj = nn.Linear(self.eff_out_dim * self.out_points * self.n_groups, self.query_dim)
        self.act = nn.ReLU(inplace=True)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.parameter_generator.weight)

    def inner_forward(self, x, query):
        B, Q, G, P, C = x.shape
        assert G == self.n_groups
        assert P == self.in_points
        assert C == self.eff_in_dim

        '''generate mixing parameters'''
        params = self.parameter_generator(query)
        params = params.reshape(B*Q, G, -1)
        out = x.reshape(B*Q, G, P, C)

        M, S = params.split([self.m_parameters, self.s_parameters], 2)
        M = M.reshape(B*Q, G, self.eff_in_dim, self.eff_out_dim)
        S = S.reshape(B*Q, G, self.out_points, self.in_points)

        '''adaptive channel mixing'''
        out = torch.matmul(out, M)
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        '''adaptive point mixing'''
        out = torch.matmul(S, out)  # implicitly transpose and matmul
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        '''linear transfomation to query dim'''
        out = out.reshape(B, Q, -1)
        out = self.out_proj(out)
        out = query + out

        return out

    def forward(self, x, query):
        if self.training and x.requires_grad:
            return cp(self.inner_forward, x, query, use_reentrant=False)
        else:
            return self.inner_forward(x, query)