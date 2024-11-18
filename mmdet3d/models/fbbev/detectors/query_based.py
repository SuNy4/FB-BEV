# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE

import torch
import torch.nn.functional as F
import torch.nn as nn
from mmcv.runner import force_fp32
import os
from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
from mmdet.models import DETECTORS
from mmdet3d.models import builder
from mmdet3d.models.detectors import CenterPoint
from mmdet3d.models.builder import build_head, build_neck
import numpy as np
import copy 
import spconv.pytorch as spconv
from tqdm import tqdm 
from mmdet3d.models.fbbev.utils import run_time
import torch
from torchvision.utils import make_grid
import torchvision
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict
from mmcv.runner import get_dist_info
from mmdet.core import reduce_mean
import mmcv
from mmdet3d.datasets.utils import nuscenes_get_rt_matrix
from mmdet3d.core.bbox import box_np_ops # , corner_to_surfaces_3d, points_in_convex_polygon_3d_jit
import time
from sklearn.cluster import KMeans


def generate_forward_transformation_matrix(bda, img_meta_dict=None):
    b = bda.size(0)
    hom_res = torch.eye(4)[None].repeat(b, 1, 1).to(bda.device)
    for i in range(b):
        hom_res[i, :3, :3] = bda[i]
    return hom_res


@DETECTORS.register_module()
class FBOCC(CenterPoint):

    def __init__(self, 
                 # BEVDet components
                 forward_projection=None,
                 img_bev_encoder_backbone=None,
                 img_bev_encoder_neck=None,

                 # Fast Instance Occ
                 pos_encoder = None,
                 img_query_cross_attn=None,
                 query_self_attn=None,
                 global_pos_self_attn_L1=None,
                 global_pos_self_attn_L2=None,
                 fcn_dw_encoder=None,
                 fcn_dh_encoder=None,
                 fcn_wh_encoder=None,
                 geometry_head=None,
                 inst_lvl_self_attn=None,

                 attn_level=None,
                 grid_config=None,
                 bev_fcn_encoder=None,
                 
                 occ_self_attn=None,
                 keypoint=None,

                 # BEVFormer components
                 backward_projection=None,

                 # FB-BEV components
                 frpn=None,

                 # depth_net
                 depth_net=None,

                 # occupancy head
                 occupancy_head=None,

                 # other settings.
                 embed_dim=None,
                 N_global_queries=None,
                 use_depth_supervision=False,
                 readd=False,
                 fix_void=False,
                 occupancy_save_path=None,
                 do_history=False,
                 interpolation_mode='bilinear',
                 history_cat_num=16,
                 history_cat_conv_out_channels=None,
                 single_bev_num_channels=80,

                  **kwargs):
        super(FBOCC, self).__init__(**kwargs)
        self.fix_void = fix_void
        self.num_levels = attn_level
        self.grid_config = grid_config
        self.occ_mesh = grid_config['x']
      
        # BEVDet init
        self.forward_projection = builder.build_neck(forward_projection) if forward_projection else None
        self.img_bev_encoder_backbone = builder.build_backbone(img_bev_encoder_backbone) if img_bev_encoder_backbone else None
        self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck) if img_bev_encoder_neck else None
        
        #FIOcc init
        self.main_queries = nn.Embedding(N_global_queries, embed_dim) if N_global_queries else None
        self.pos_encoder = builder.build_neck(pos_encoder) if pos_encoder else None
        self.img_query_cross_attn = builder.build_neck(img_query_cross_attn) if img_query_cross_attn else None
        self.query_self_attn = builder.build_neck(query_self_attn) if query_self_attn else None
        self.global_pos_self_attn_L1 = builder.build_neck(global_pos_self_attn_L1) if global_pos_self_attn_L1 else None
        self.global_pos_self_attn_L2 = builder.build_neck(global_pos_self_attn_L2) if global_pos_self_attn_L2 else None

        self.fcn_dw_encoder = builder.build_neck(fcn_dw_encoder) if fcn_dw_encoder else None
        self.fcn_dh_encoder = builder.build_neck(fcn_dh_encoder) if fcn_dh_encoder else None
        self.fcn_wh_encoder = builder.build_neck(fcn_wh_encoder) if fcn_wh_encoder else None

        self.bev_fcn_encoder = builder.build_neck(bev_fcn_encoder) if bev_fcn_encoder else None
        self.geometry_head = builder.build_neck(geometry_head) if geometry_head else None

   

        # FC layer


        # BEVFormer init
        self.backward_projection = builder.build_head(backward_projection) if backward_projection else None
    
        # FB-BEV init
        if not self.forward_projection: assert not frpn, 'frpn relies on LSS'
        self.frpn = builder.build_head(frpn) if frpn else None

        # Depth Net
        self.depth_net = builder.build_head(depth_net) if depth_net else None

        # Occupancy Head
        self.occupancy_head = builder.build_head(occupancy_head) if occupancy_head else None


        self.readd = readd # fuse voxel features and bev features
        
        self.use_depth_supervision = use_depth_supervision
        
        self.occupancy_save_path = occupancy_save_path # for saving data\for submitting to test server

        # # Deal with history
        # self.single_bev_num_channels = single_bev_num_channels
        # self.do_history = do_history
        # self.interpolation_mode = interpolation_mode
        # self.history_cat_num = history_cat_num
        # self.history_cam_sweep_freq = 0.5 # seconds between each frame
        # history_cat_conv_out_channels = (history_cat_conv_out_channels 
        #                                  if history_cat_conv_out_channels is not None 
        #                                  else self.single_bev_num_channels)
        # ## Embed each sample with its relative temporal offset with current timestep
        # conv = nn.Conv2d if self.forward_projection.nx[-1] == 1 else nn.Conv3d
        # self.history_keyframe_time_conv = nn.Sequential(
        #      conv(self.single_bev_num_channels + 1,
        #              self.single_bev_num_channels,
        #              kernel_size=1,
        #              padding=0,
        #              stride=1),
        #      nn.SyncBatchNorm(self.single_bev_num_channels),
        #      nn.ReLU(inplace=True))
        # ## Then concatenate and send them through an MLP.
        # self.history_keyframe_cat_conv = nn.Sequential(
        #     conv(self.single_bev_num_channels * (self.history_cat_num + 1),
        #             history_cat_conv_out_channels,
        #             kernel_size=1,
        #             padding=0,
        #             stride=1),
        #     nn.SyncBatchNorm(history_cat_conv_out_channels),
        #     nn.ReLU(inplace=True))
        # self.history_sweep_time = None
        self.history_bev = None
        # self.history_bev_before_encoder = None
        # self.history_seq_ids = None
        # self.history_forward_augs = None
        # self.count = 0

    def with_specific_component(self, component_name):
        """Whether the model owns a specific component"""
        return getattr(self, component_name, None) is not None
    
    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
      
        x = self.img_backbone(imgs)
       
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
      
        return x

    @force_fp32()
    def bev_encoder(self, x):
        if self.with_specific_component('img_bev_encoder_backbone'):
            x = self.img_bev_encoder_backbone(x)
        
        if self.with_specific_component('img_bev_encoder_neck'):
            x = self.img_bev_encoder_neck(x)
        
        if type(x) not in [list, tuple]:
             x = [x]

        return x

    @force_fp32()
    def fuse_history(self, curr_bev, img_metas, bda): # align features with 3d shift

        voxel_feat = True  if len(curr_bev.shape) == 5 else False
        if voxel_feat:
            curr_bev = curr_bev.permute(0, 1, 4, 2, 3) # n, c, z, h, w
        
        seq_ids = torch.LongTensor([
            single_img_metas['sequence_group_idx'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        start_of_sequence = torch.BoolTensor([
            single_img_metas['start_of_sequence'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        forward_augs = generate_forward_transformation_matrix(bda)

        curr_to_prev_ego_rt = torch.stack([
            single_img_metas['curr_to_prev_ego_rt']
            for single_img_metas in img_metas]).to(curr_bev)

        ## Deal with first batch
        if self.history_bev is None:
            self.history_bev = curr_bev.clone()
            self.history_seq_ids = seq_ids.clone()
            self.history_forward_augs = forward_augs.clone()

            # Repeat the first frame feature to be history
            if voxel_feat:
                self.history_bev = curr_bev.repeat(1, self.history_cat_num, 1, 1, 1) 
            else:
                self.history_bev = curr_bev.repeat(1, self.history_cat_num, 1, 1)
            # All 0s, representing current timestep.
            self.history_sweep_time = curr_bev.new_zeros(curr_bev.shape[0], self.history_cat_num)


        self.history_bev = self.history_bev.detach()

        assert self.history_bev.dtype == torch.float32

        ## Deal with the new sequences
        # First, sanity check. For every non-start of sequence, history id and seq id should be same.

        assert (self.history_seq_ids != seq_ids)[~start_of_sequence].sum() == 0, \
                "{}, {}, {}".format(self.history_seq_ids, seq_ids, start_of_sequence)

        ## Replace all the new sequences' positions in history with the curr_bev information
        self.history_sweep_time += 1 # new timestep, everything in history gets pushed back one.
        if start_of_sequence.sum()>0:
            if voxel_feat:    
                self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1, 1)
            else:
                self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1)
            
            self.history_sweep_time[start_of_sequence] = 0 # zero the new sequence timestep starts
            self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
            self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]


        ## Get grid idxs & grid2bev first.
        if voxel_feat:
            n, c_, z, h, w = curr_bev.shape

        # Generate grid
        xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w, 1).expand(h, w, z)
        ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1, 1).expand(h, w, z)
        zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 1, z).expand(h, w, z)
        grid = torch.stack(
            (xs, ys, zs, torch.ones_like(xs)), -1).view(1, h, w, z, 4).expand(n, h, w, z, 4).view(n, h, w, z, 4, 1)

        # This converts BEV indices to meters
        # IMPORTANT: the feat2bev[0, 3] is changed from feat2bev[0, 2] because previous was 2D rotation
        # which has 2-th index as the hom index. Now, with 3D hom, 3-th is hom
        feat2bev = torch.zeros((4,4),dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.forward_projection.dx[0]
        feat2bev[1, 1] = self.forward_projection.dx[1]
        feat2bev[2, 2] = self.forward_projection.dx[2]
        feat2bev[0, 3] = self.forward_projection.bx[0] - self.forward_projection.dx[0] / 2.
        feat2bev[1, 3] = self.forward_projection.bx[1] - self.forward_projection.dx[1] / 2.
        feat2bev[2, 3] = self.forward_projection.bx[2] - self.forward_projection.dx[2] / 2.
        # feat2bev[2, 2] = 1
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1,4,4)

        ## Get flow for grid sampling.
        # The flow is as follows. Starting from grid locations in curr bev, transform to BEV XY11,
        # backward of current augmentations, curr lidar to prev lidar, forward of previous augmentations,
        # transform to previous grid locations.
        rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_ego_rt
                   @ torch.inverse(forward_augs) @ feat2bev)

        grid = rt_flow.view(n, 1, 1, 1, 4, 4) @ grid

        # normalize and sample
        normalize_factor = torch.tensor([w - 1.0, h - 1.0, z - 1.0], dtype=curr_bev.dtype, device=curr_bev.device)
        grid = grid[:,:,:,:, :3,0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0
        

        tmp_bev = self.history_bev
        if voxel_feat: 
            n, mc, z, h, w = tmp_bev.shape
            tmp_bev = tmp_bev.reshape(n, mc, z, h, w)
        sampled_history_bev = F.grid_sample(tmp_bev, grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4), align_corners=True, mode=self.interpolation_mode)

        ## Update history
        # Add in current frame to features & timestep
        self.history_sweep_time = torch.cat(
            [self.history_sweep_time.new_zeros(self.history_sweep_time.shape[0], 1), self.history_sweep_time],
            dim=1) # B x (1 + T)

        if voxel_feat:
            sampled_history_bev = sampled_history_bev.reshape(n, mc, z, h, w)
            curr_bev = curr_bev.reshape(n, c_, z, h, w)
        feats_cat = torch.cat([curr_bev, sampled_history_bev], dim=1) # B x (1 + T) * 80 x H x W or B x (1 + T) * 80 xZ x H x W 

        # Reshape and concatenate features and timestep
        feats_to_return = feats_cat.reshape(
                feats_cat.shape[0], self.history_cat_num + 1, self.single_bev_num_channels, *feats_cat.shape[2:]) # B x (1 + T) x 80 x H x W
        if voxel_feat:
            feats_to_return = torch.cat(
            [feats_to_return, self.history_sweep_time[:, :, None, None, None, None].repeat(
                1, 1, 1, *feats_to_return.shape[3:]) * self.history_cam_sweep_freq
            ], dim=2) # B x (1 + T) x 81 x Z x H x W
        else:
            feats_to_return = torch.cat(
            [feats_to_return, self.history_sweep_time[:, :, None, None, None].repeat(
                1, 1, 1, feats_to_return.shape[3], feats_to_return.shape[4]) * self.history_cam_sweep_freq
            ], dim=2) # B x (1 + T) x 81 x H x W

        # Time conv
        feats_to_return = self.history_keyframe_time_conv(
            feats_to_return.reshape(-1, *feats_to_return.shape[2:])).reshape(
                feats_to_return.shape[0], feats_to_return.shape[1], -1, *feats_to_return.shape[3:]) # B x (1 + T) x 80 xZ x H x W

        # Cat keyframes & conv
        feats_to_return = self.history_keyframe_cat_conv(
            feats_to_return.reshape(
                feats_to_return.shape[0], -1, *feats_to_return.shape[3:])) # B x C x H x W or B x C x Z x H x W
        
        self.history_bev = feats_cat[:, :-self.single_bev_num_channels, ...].detach().clone()
        self.history_sweep_time = self.history_sweep_time[:, :-1]
        self.history_forward_augs = forward_augs.clone()
        if voxel_feat:
            feats_to_return = feats_to_return.permute(0, 1, 3, 4, 2)
        if not self.do_history:
            self.history_bev = None
        return feats_to_return.clone()

#######################################################################################################
    ### OVerall Flow: 2D feature encode with global pos info -> 2D 3D deformable attention(Sim Loss) -> 3D deformable self attention -> Occupied MLP head(CE loss) -> Sparsified voxels(large norm self attention) -> MLP head -> Combine Map
    def extract_img_bev_feat(self, img, img_metas, **kwargs):
        """Extract features of images."""

        return_map = {}

        context = self.image_encoder(img[0]).permute(0, 1, 2, 4, 3) # bs, Ncam, C, W, H

        cam_params = img[1:7] #rot, tran, intrin, post_rot, post_tran, bda: *cam_params
        
        # Local Image Pos Encode
        if self.with_specific_component('pos_encoder'):
            img_local_pos_encode = self.pos_encoder(context, *cam_params, mode='Local') # bs, Ncam, WH, num_freqs*4
            img_local_pos_encode = img_local_pos_encode.flatten(0, 1) # bsNcam, WH, num_freqs*4

        if self.with_specific_component('img_query_cross_attn'):
            bs, Ncam, C, W, H = context.shape

            context = context.flatten(-2, -1)
            context = context.flatten(0, 1).permute(0, 2, 1) # bsNcam, WH, C
            general_queries = self.main_queries.weight
            global_queries = general_queries.unsqueeze(0).repeat(bs*Ncam, 1, 1) # bsNcam, N, C

            value = torch.cat([context, img_local_pos_encode], dim=-1)
            input_c = value.shape[2]
            fc_layer_1 = nn.Sequential(
                nn.Linear(input_c, input_c*2),
                nn.GELU(),
                nn.Linear(input_c*2, C)
            ).to('cuda')
            value = fc_layer_1(value)

            # Find instance unrelated to img position, only value have image position info, since queries are general
            inst_queries, attn_weights = self.img_query_cross_attn(
                query = global_queries,
                key = context,
                value = value
            )
            # inst_queries: bs*Ncam, N_queries, C
            # attn_weights: bs*Ncam, N_queries, WH
            _, max_indices = torch.max(attn_weights, dim=-1) # bs*Ncam, N_queries
            img_w = max_indices // H # bs*Ncam, N_queries
            img_h = max_indices % H # bs*Ncam, N_queries
            max_img_coords = torch.cat([img_w.unsqueeze(-1), img_h.unsqueeze(-1)], dim = -1) # bs*Ncam, N_queries, 2
            max_img_coords = max_img_coords.reshape(bs, Ncam, -1, 2) # bs, Ncam, N_queries, 2
            query_global_pos_encode, sph_coords = self.pos_encoder(max_img_coords, *cam_params, mode='Global', custum_input=True, height=H, width=W)
            query_global_pos_encode = query_global_pos_encode.flatten(0, 1)
            sph_coords = sph_coords.flatten(1, 2)
            # query global pos encoder: bsNcam, N_queries, num_freqs*4
            # sph_coords: bs, Ncam*N_queries, 2 (theta, phi)

        if self.with_specific_component('query_self_attn'):
            inst_queries, _ = self.query_self_attn(
                query = inst_queries
            ) # bsNCam, N_queries, C

        if self.with_specific_component('global_pos_self_attn_L1'):
            
            value = torch.cat([inst_queries, query_global_pos_encode], dim=-1)
            input_c = value.shape[2]
            fc_layer_2 = nn.Sequential(
                nn.Linear(input_c, input_c*2),
                nn.GELU(),
                nn.Linear(input_c*2, C)
            ).to('cuda')
            value = fc_layer_2(value)

            inst_queries, _ = self.global_pos_self_attn_L1(
                query = inst_queries,
                key = inst_queries,
                value = value
            ) # bsNCam, N_queries, C

        if self.with_specific_component('global_pos_self_attn_L2'):
            inst_queries, _ = self.global_pos_self_attn_L2(
                query = inst_queries,
            ) # bsNCam, N_queries, C

        # if self.with_specific_component('radius_mlp'):
            _, _, input_c = inst_queries.shape
            inst_queries = inst_queries.reshape(bs, Ncam, -1, input_c) # bs, Ncam, N_queries, C
            inst_queries = inst_queries.flatten(1, 2) # bs, Ncam*N_queries, C
            
            fc_layer_3 = nn.Sequential(
                nn.Linear(input_c, input_c*2),
                nn.GELU(),
                nn.Linear(input_c*2, 100)
            ).to('cuda')
            
            logits = fc_layer_3(inst_queries) # bs, Ncam*N_queries, 100
            r_prob = F.softmax(logits, dim=-1) # bs, Ncam*N_queries, 100
            
            r = torch.argmax(r_prob, dim=-1).unsqueeze(-1) * 0.4 # bs, Ncam*N_queries, 1
            # sph_coords = torch.cat(r_prob, sph_coords, dim=-1) # bs, Ncam*N_quries, 3 (r, theta, phi)
            theta = sph_coords[..., 0].unsqueeze(-1)
            phi = sph_coords[..., 1].unsqueeze(-1)
            
            ###Spherical to Cartisian idx###
            D = r * torch.cos(phi) * torch.cos(theta) - self.grid_config['x'][0]  
            W = r * torch.cos(phi) * torch.sin(theta) - self.grid_config['y'][0] 
            H = r * torch.sin(phi) - self.grid_config['z'][0] 
            
            D = torch.clamp((D / 0.4).long(), 0, 199) # bs, Ncam*N_queries, 1
            W = torch.clamp((W / 0.4).long(), 0, 199) # bs, Ncam*N_queries, 1
            H = torch.clamp((H / 0.4).long(), 0, 15) # bs, Ncam*N_queries, 1

            cart_idx = torch.cat([D, W, H], dim=-1) # bs, Ncam*N_queries, 3

        #if self.with_specific_component('TPV_FCN'):
            D = 200
            W = 200
            H = 16
            D = torch.linspace(self.grid_config['x'][0], self.grid_config['x'][1], D).to('cuda')
            W = torch.linspace(self.grid_config['y'][0], self.grid_config['y'][0], W).to('cuda')
            H = torch.linspace(self.grid_config['z'][0], self.grid_config['z'][0], H).to('cuda')

            d, w = torch.meshgrid(D, W, indexing='ij')
            dw_plane = torch.stack([d, w], dim=-1)# 200x200x2
            dw_plane = dw_plane[None, :].repeat(bs, 1, 1, 1)     

            d, h = torch.meshgrid(D, H, indexing='ij')
            dh_plane = torch.stack([d, h], dim=-1)
            dh_plane = dh_plane[None, :].repeat(bs, 1, 1, 1)

            w, h = torch.meshgrid(W, H, indexing='ij')
            wh_plane = torch.stack([w, h], dim=-1)
            wh_plane = wh_plane[None, :].repeat(bs, 1, 1, 1)

            dw_plane = self.pos_encoder(dw_plane, map_input=True)
            dh_plane = self.pos_encoder(dh_plane, map_input=True)
            wh_plane = self.pos_encoder(wh_plane, map_input=True)

            zero_pad = torch.zeros(bs, 200, 200, 96).to(dw_plane.device)
            dw_plane = torch.cat([zero_pad, dw_plane], dim=-1)
            zero_pad = torch.zeros(bs, 200, 16, 96).to(dh_plane.device)
            dh_plane = torch.cat([zero_pad, dh_plane], dim=-1)
            wh_plane = torch.cat([zero_pad, wh_plane], dim=-1)

            for i, index in enumerate(cart_idx):
                d_indices = index[:, 0]
                w_indices = index[:, 1]
                h_indices = index[:, 2]
                dw_plane[i, d_indices, w_indices] = inst_queries[i, :len(index)]
                dh_plane[i, d_indices, h_indices] = inst_queries[i, :len(index)]
                wh_plane[i, w_indices, h_indices] = inst_queries[i, :len(index)]

            dw_plane = self.fcn_dw_encoder(dw_plane.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).unsqueeze(3)
            dh_plane = self.fcn_dh_encoder(dh_plane.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).unsqueeze(2)
            wh_plane = self.fcn_wh_encoder(wh_plane.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).unsqueeze(1)

            occ_feat = dw_plane * dh_plane * wh_plane
            bs, D, W, H, _ = occ_feat.shape
            input_reshaped = occ_feat.view(bs, D * W * H, C)  # bs, D*W*H, C
            query_reshaped = general_queries.t()  # C, N
            occ_feat = torch.bmm(input_reshaped, query_reshaped.unsqueeze(0).expand(bs, -1, -1))  # bs, D*W*H, N
            # occ_feat = occ_feat.view(bs, D, W, H, -1).sigmoid()
            
            threshold = 0.4
            mask = (occ_feat.clone().sigmoid() <= threshold).all(dim=-1)
            
            occ_feat[mask] = 0 
            occ_feat = occ_feat.reshape(bs, D, W, H, -1) # bs, D, W, H, 50
            return_map['sparse_feat'] = occ_feat
            return_map['sparse_idx'] = None

            # occ_index = [] # [bs][N][3] occupied voxel indexes
            # max_len = 0
            # for i, per_batch in enumerate(occ_feat):
            #     index_per_batch = per_batch.nonzero().squeeze(-1)
            #     occ_index.append(index_per_batch)
            #     max_len = max(max_len, len(index_per_batch))

            # sparse_voxel = torch.zeros([
            #     bs, max_len, 50
            # ]).cuda()

            # for i, index in enumerate(occ_index):   
            #     d_indices = index[:, 0]
            #     w_indices = index[:, 1]
            #     h_indices = index[:, 2]
            #     sparse_voxel[i, :len(index)] = occ_feat[i, d_indices, w_indices, h_indices]

            # return_map['geom'] = ~mask
            # return_map['sparse_idx'] = occ_index
            # return_map['sparse_feat'] = sparse_voxel

#######################################################################################################

        # Fuse History
        # bev_feat = self.fuse_history(bev_feat, img_metas, img[6])
        
        #bev_feat = self.bev_encoder(bev_feat)
        # return_map['img_bev_feat'] = output

        return return_map

    def extract_lidar_bev_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""

        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        bev_feat = self.pts_middle_encoder(voxel_features, coors, batch_size)
        bev_feat = self.bev_encoder(bev_feat)
        return bev_feat

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        results={}
        if img is not None and self.with_specific_component('image_encoder'):
            results.update(self.extract_img_bev_feat(img, img_metas, **kwargs))
        if points is not None and self.with_specific_component('pts_voxel_encoder'):
            results['lidar_bev_feat'] = self.extract_lidar_bev_feat(points, img, img_metas)

        return results


    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      gt_occupancy_flow=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """


        results= self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        losses = dict()

        if  self.with_pts_bbox:
            losses_pts = self.forward_pts_train(results['img_bev_feat'], gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
            losses.update(losses_pts)
            
        if self.with_specific_component('occupancy_head'):
            losses_occupancy = self.occupancy_head.forward_train(results['sparse_feat'], results=results, gt_occupancy=kwargs['gt_occupancy'], cam_mask=kwargs['cam_visible_mask'], gt_occupancy_flow=gt_occupancy_flow, sparse=True)
            losses.update(losses_occupancy)

        if self.with_specific_component('frpn'):
            losses_mask = self.frpn.get_bev_mask_loss(kwargs['gt_bev_mask'], results['bev_mask_logit'])
            losses.update(losses_mask)

        if self.use_depth_supervision and self.with_specific_component('depth_net'):
            loss_depth = self.depth_net.get_depth_loss(kwargs['gt_depth'], results['depth'])
            losses.update(loss_depth)

        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        self.do_history = True
        if img_inputs is not None:
            for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
                if not isinstance(var, list) :
                    raise TypeError('{} must be a list, but got {}'.format(
                        name, type(var)))        
            num_augs = len(img_inputs)
            if num_augs != len(img_metas):
                raise ValueError(
                    'num of augmentations ({}) != num of image meta ({})'.format(
                        len(img_inputs), len(img_metas)))

            if num_augs==1 and not img_metas[0][0].get('tta_config', dict(dist_tta=False))['dist_tta']:
                return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                    **kwargs)
            else:
                return self.aug_test(points, img_metas, img_inputs, **kwargs)
        
        elif points is not None:
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                    **kwargs)
        
    def aug_test(self,points,
                    img_metas,
                    img_inputs=None,
                    visible_mask=[None],
                    **kwargs):
        """Test function without augmentaiton."""
        assert False
        return None

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    visible_mask=[None],
                    return_raw_occ=False,
                    **kwargs):
        """Test function without augmentaiton."""
        results = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        

        bbox_list = [dict() for _ in range(len(img_metas))]
        
        if  self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(results['img_bev_feat'], img_metas, rescale=rescale)
        else:
            bbox_pts = [None for _ in range(len(img_metas))]


        if self.with_specific_component('occupancy_head'):
            # t=time.time()
            pred_occupancy = self.occupancy_head.forward_sparse(results['sparse_feat'], results['sparse_idx'], **kwargs)['output_voxels'][0]
            # print(f"\nHead time:{time.time()-t}")

            pred_occupancy = pred_occupancy.permute(0, 2, 3, 4, 1)[0]
            # if self.fix_void:
            #     pred_occupancy = pred_occupancy[..., 1:]     
            pred_occupancy = pred_occupancy.softmax(-1)


            # convert to CVPR2023 Format
            pred_occupancy = pred_occupancy.permute(3, 2, 0, 1)
            pred_occupancy = torch.flip(pred_occupancy, [2])
            pred_occupancy = torch.rot90(pred_occupancy, -1, [2, 3])
            pred_occupancy = pred_occupancy.permute(2, 3, 1, 0)
            
            if return_raw_occ:
                pred_occupancy_category = pred_occupancy
            else:
                pred_occupancy_category = pred_occupancy.argmax(-1)
            t=time.time()

            # # do not change the order
            # if self.occupancy_save_path is not None:
            #     scene_name = img_metas[0]['scene_name']
            #     sample_token = img_metas[0]['sample_idx']
            #     mask_camera = visible_mask[0][0]
            #     masked_pred_occupancy = pred_occupancy[mask_camera].cpu().numpy()
            #     save_path = os.path.join(self.occupancy_save_path, 'occupancy_pred', scene_name+'_'+sample_token)
            #     np.savez_compressed(save_path, pred=masked_pred_occupancy, sample_token=sample_token) 


            # For test server
            if self.occupancy_save_path is not None:
                    scene_name = img_metas[0]['scene_name']
                    sample_token = img_metas[0]['sample_idx']
                    # mask_camera = visible_mask[0][0]
                    # masked_pred_occupancy = pred_occupancy[mask_camera].cpu().numpy()
                    save_pred_occupancy = pred_occupancy.argmax(-1).cpu().numpy()
                    save_path = os.path.join(self.occupancy_save_path, 'occupancy_pred', f'{sample_token}.npz')
                    np.savez_compressed(save_path, save_pred_occupancy.astype(np.uint8)) 

            pred_occupancy_category= pred_occupancy_category.cpu().numpy()

        else:
            pred_occupancy_category =  None

        if results.get('bev_mask_logit', None) is not None:
            pred_bev_mask = results['bev_mask_logit'].sigmoid() > 0.5
            iou = IOU(pred_bev_mask.reshape(1, -1), kwargs['gt_bev_mask'][0].reshape(1, -1)).cpu().numpy()
        else:
            iou = None

        assert len(img_metas) == 1
        for i, result_dict in enumerate(bbox_list):
            result_dict['pts_bbox'] = bbox_pts[i]
            result_dict['iou'] = iou
            result_dict['pred_occupancy'] = pred_occupancy_category
            result_dict['index'] = img_metas[0]['index']
        return bbox_list, t

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        results = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        assert self.with_pts_bbox
        outs = self.pts_bbox_head(results['img_bev_feat'])
        return outs
