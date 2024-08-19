# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import BaseModule, auto_fp16
from mmdet.models import NECKS

import torch
import torch.nn as nn

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
    def __init__(self, scale_factor, mode, in_channels, out_channels):
        super(UpsampleLayer, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=True)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.gelu = nn.GELU()

    def forward(self, x):
        out = self.upsample(x)
        out = self.conv(out)
        return out
    
@NECKS.register_module()
class BEV2DFCN(nn.Module):
    def __init__(self, flatten_height, height, in_channels, out_channels):
        super(BEV2DFCN, self).__init__()
        self.flatten_height = flatten_height
        self.in_channels = in_channels
        # self.conv0 = nn.Conv2d(self.in_channels*height, self.in_channels, kernel_size=1)
        self.conv1 = nn.Conv2d(self.in_channels*height, out_channels, kernel_size=1)
        # self.bn_flat0 = nn.BatchNorm2d(self.in_channels*height/2)
        self.bn_flat1 = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.decoder1 = UpsampleLayer(2, 'bilinear', out_channels, out_channels)
        self.decoder2 = UpsampleLayer(2, 'bilinear', out_channels, out_channels)
        # self.encoder1 = AggregationBlock(out_channels, out_channels*2)
        # self.encoder2 = AggregationBlock(out_channels*2, out_channels*4)
        
        # self.decoder1 = nn.ConvTranspose2d(out_channels*4, out_channels*2, padding=1, kernel_size=4, stride=2)
        # self.decoder2 = nn.ConvTranspose2d(out_channels*2, out_channels, padding=1, kernel_size=4, stride=2)
        
        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.flatten_height:
            # x = self.bn_flat0(self.conv0(x))
            x = self.bn_flat1(self.conv1(x))
            x = self.gelu(x)
        
        # Downsample
        # e1 = self.encoder1(x)
        # e2 = self.encoder2(e1)
        e1 = self.maxpool(x)
        e2 = self.maxpool(e1)

        # Upsample
        d1 = self.gelu(self.bn1(self.decoder1(e2)))
        d1 = d1 + e1
        
        d2 = self.gelu(self.bn2(self.decoder2(d1)))
        out = d2 + x

        return out