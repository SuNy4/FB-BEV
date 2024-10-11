# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
import torch.utils.checkpoint as cp
from mmdet.models import NECKS
from mmcv.ops import MultiScaleDeformableAttention
from mmcv.runner import force_fp32, auto_fp16
from .deform_squeeze import DeformableSqueezeAttention
import math

class SphericalPositionalEncodingWithView(torch.nn.Module):
    def __init__(self, max_radius=40, num_freqs=8):
        super(SphericalPositionalEncodingWithView, self).__init__()
        self.num_freqs = num_freqs
        self.max_radius = max_radius
        self.freq_bands = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs).cuda()
    
    def forward(self, coords, encode_theta=False, theta_only=False):
        """
        Args:
            coords: Tensor of shape (N, 3) representing (r, theta, phi) coordinates in 3D space.

        Returns:
            pos_enc: (N, num_freqs * 6)
        """
        if theta_only:
            theta = coords[..., 1]
            encoded_theta = torch.cat([torch.sin(self.freq_bands[None, :] * theta[:, None]),
                                       torch.cos(self.freq_bands[None, :] * theta[:, None])], dim=-1)
            return encoded_theta
    
        else:
            r, theta, phi, h = coords[..., 0], coords[..., 1], coords[..., 2], coords[..., 3]
            r_max = r.max()

            r_normalized = r / (r_max + 1e-8)
            encoded_r = torch.cat([torch.sin(self.freq_bands[None, :] * r_normalized[:, None]),
                                   torch.cos(self.freq_bands[None, :] * r_normalized[:, None])], dim=-1)

            encoded_phi = torch.cat([torch.sin(self.freq_bands[None, :] * phi[:, None]),
                                     torch.cos(self.freq_bands[None, :] * phi[:, None])], dim=-1)
            
            encoded_h = torch.cat([torch.sin(self.freq_bands[None, :] * h[:, None]),
                                   torch.cos(self.freq_bands[None, :] * h[:, None])], dim=-1)
            if encode_theta:
                encoded_theta = torch.cat([torch.sin(self.freq_bands[None, :] * theta[:, None]),
                                           torch.cos(self.freq_bands[None, :] * theta[:, None])], dim=-1)
        
                pos_enc = torch.cat([encoded_r, encoded_phi, encoded_h, encoded_theta], dim=-1).cuda()
                return pos_enc
            
            pos_enc = torch.cat([encoded_r, encoded_phi, encoded_h], dim=-1).cuda()
            return pos_enc
        
############################# Using Local pos encode
@NECKS.register_module()
class PosDeformableTransformerLayer(nn.Module):

    def __init__(self,
                 embed_dims,
                 num_heads=8,
                 num_levels=3,
                 num_points=4,
                 mlp_ratio=4,
                 grid_config=None,
                 data_config=None,
                 pos_encoder=SphericalPositionalEncodingWithView,
                 attn_layer=MultiScaleDeformableAttention,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()
        self.num_levels=num_levels
        self.cam_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 0)]
        self.x_bound = grid_config['x']
        self.y_bound = grid_config['y']
        self.z_bound = grid_config['z']
        self.cam_theta = grid_config['Cam_Setting']
        self.embed_dims = embed_dims
        # self.norm1 = norm_layer(embed_dims)
        self.pos_encoder = pos_encoder(num_freqs = embed_dims // 8)
        if isinstance(attn_layer, str):
            if attn_layer == 'DeformableSqueezeAttention':
                attn_layer = DeformableSqueezeAttention
            elif attn_layer == 'MultiScaleDeformableAttention':
                attn_layer = MultiScaleDeformableAttention
            else:
                raise ValueError(f"Unknown attention layer: {attn_layer}")
        self.attn = attn_layer(
            embed_dims, num_heads, num_levels, num_points, batch_first=True, im2col_step=256)
        if mlp_ratio == 0:
            return
        self.original_dim=data_config['input_size']
        self.norm2 = norm_layer(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dims * mlp_ratio, embed_dims),
        )

    def get_reference_points(self, coords=None, dim='3d', device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        torch.autograd.set_detect_anomaly(True)
        if dim == '2d':

            x_range = self.x_bound[1] - self.x_bound[0]
            y_range = self.y_bound[1] - self.y_bound[0]
            z_range = self.z_bound[1] - self.z_bound[0]

            coords[..., 0] = coords[..., 0] * x_range + self.x_bound[0]
            coords[..., 1] = coords[..., 1] * y_range + self.y_bound[0]
            coords[..., 2] = coords[..., 2] * z_range + self.z_bound[0]

            world_coords = coords

            return world_coords
                # reference points in 3D space, used in spatial cross-attention (SCA)

        if dim == '3d':

            X = torch.arange(*self.x_bound, dtype=torch.float) # + self.x_bound[-1]/2
            Y = torch.arange(*self.y_bound, dtype=torch.float) # + self.y_bound[-1]/2
            Z = torch.arange(*self.z_bound, dtype=torch.float) # + self.z_bound[-1]/2
            Y, X, Z = torch.meshgrid([Y, X, Z])
            coords = torch.stack([X, Y, Z], dim=-1)
            coords = coords.to(dtype).to(device)

            return coords
        
    def cartesian_to_spherical(self, grid_coords):
        x = grid_coords[..., 0] # D
        y = grid_coords[..., 1] # W
        z = grid_coords[..., 2] # H
        
        r = torch.sqrt(x**2 + y**2 + z**2)  # Distance from the origin
        theta = torch.atan2(y, x)  # Angle in the xy-plane
        phi = torch.atan2(z, torch.sqrt(x**2 + y**2))  # Angle with respect to z-axis
        
        # Stack spherical coordinates
        spherical_coords = torch.stack([r, theta, phi], dim=-1)
        return spherical_coords
    
    @force_fp32(apply_to=('reference_points', 'cam_params'))
    def point_sampling(self, reference_points, pc_range=None,  
                       img_metas=None, cam_params=None, gt_bboxes_3d=None):

        rots, trans, intrins, post_rots, post_trans, bda = cam_params
        B, N, _ = trans.shape
        eps = 1e-5
        ogfH, ogfW = self.original_dim
        reference_points = reference_points[None, None].repeat(B, N, 1, 1, 1, 1)
        reference_points = torch.inverse(bda).view(B, 1, 1, 1, 1, 3,
                          3).matmul(reference_points.unsqueeze(-1)).squeeze(-1)
        reference_points -= trans.view(B, N, 1, 1, 1, 3)
        combine = rots.matmul(torch.inverse(intrins)).float().inverse()
        reference_points_cam = combine.view(B, N, 1, 1, 1, 3, 3).matmul(reference_points.unsqueeze(-1)).squeeze(-1)
        reference_points_cam = torch.cat([reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps),  reference_points_cam[..., 2:3]], 5
            )
        reference_points_cam = post_rots.view(B, N, 1, 1, 1, 3, 3).matmul(reference_points_cam.unsqueeze(-1)).squeeze(-1)
        reference_points_cam += post_trans.view(B, N, 1, 1, 1, 3) 
        reference_points_cam[..., 0] /= ogfW
        reference_points_cam[..., 1] /= ogfH
        mask = (reference_points_cam[..., 2:3] > eps)
        mask = (mask & (reference_points_cam[..., 0:1] > eps) 
                 & (reference_points_cam[..., 0:1] < (1.0-eps)) 
                 & (reference_points_cam[..., 1:2] > eps) 
                 & (reference_points_cam[..., 1:2] < (1.0-eps)))
        B, N, D, W, H, _ = reference_points_cam.shape
        reference_points_cam = reference_points_cam.permute(1, 0, 2, 3, 4, 5) #.reshape(N, B, D, W, H, 3)
        mask = mask.permute(1, 0, 2, 3, 4, 5).squeeze(-1) #.reshape(N, B, D, W, H, 1).squeeze(-1)

        return reference_points, reference_points_cam[..., :2], mask, reference_points_cam[..., 2:3]
    
    @force_fp32(apply_to=('reference_points', 'cam_params'))
    def queries_point_sampling(self, reference_points, pc_range=None,  
                       img_metas=None, cam_params=None, gt_bboxes_3d=None):
        rots, trans, intrins, post_rots, post_trans, bda = cam_params
        B, N, _ = trans.shape
        eps = 1e-5
        ogfH, ogfW = self.original_dim

        reference_points = reference_points.unsqueeze(1).repeat(1, N, 1, 1)
        reference_points = torch.inverse(bda).view(B, 1, 1, 3,
                          3).matmul(reference_points.unsqueeze(-1)).squeeze(-1)
        reference_points = reference_points - trans.view(B, N, 1, 3)
        combine = rots.matmul(torch.inverse(intrins)).float().inverse()
        reference_points_cam = combine.view(B, N, 1, 3, 3).matmul(reference_points.unsqueeze(-1)).float().squeeze(-1)

        reference_points_cam = torch.cat([reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps),  reference_points_cam[..., 2:3]], dim=3
            )
        
        reference_points_cam = post_rots.view(B, N, 1, 3, 3).matmul(reference_points_cam.unsqueeze(-1)).float().squeeze(-1)
        reference_points_cam += post_trans.view(B, N, 1, 3) 

        reference_points_cam[..., 0] /= ogfW
        reference_points_cam[..., 1] /= ogfH

        mask = (reference_points_cam[..., 2:3] > eps)
        mask = (mask & (reference_points_cam[..., 0:1] > eps) 
                 & (reference_points_cam[..., 0:1] < (1.0-eps)) 
                 & (reference_points_cam[..., 1:2] > eps) 
                 & (reference_points_cam[..., 1:2] < (1.0-eps)))
        B, N, N_queries, _ = reference_points_cam.shape
        reference_points_cam = reference_points_cam.permute(1, 0, 2, 3).reshape(N, B, N_queries, 3)
        mask = mask.permute(1, 0, 2, 3).reshape(N, B, N_queries, 1).squeeze(-1)

        return reference_points, reference_points_cam[..., :2], mask, reference_points_cam[..., 2:3]
    
    @force_fp32(apply_to=('value', 'query_pos'))
    def forward(self,
                query=None,
                value=None,
                query_pos=None,
                ref_pts=None,
                view_transform=False,
                bev_value=False,
                occ_value=False,
                spatial_shapes=None,
                level_start_index=None,
                cam_params=None,
                ):

        # View Transformation
        bs = value.shape[0] # bs, Ncam, C, H, W

        if view_transform:
            indexes = [[] for _ in range(bs)]
            overlapped_indexes = []
            pos_encode_indexes = [[] for _ in range(bs)]

            # D, W, H, 3 (x, y, z cartesian)
            wrld_ref_3d = self.get_reference_points(
                dim='3d', 
                device='cuda', 
                dtype=torch.float
                )
            D, W, H, _ = wrld_ref_3d.shape
            
            # D, W, H, 4 (r, theta, phi, h: spherical) for pos encode
            h = wrld_ref_3d[..., -1].unsqueeze(-1)
            sph_ref_pts = torch.cat([self.cartesian_to_spherical(wrld_ref_3d), h], dim=-1)

            global_pos_encode = self.pos_encoder(sph_ref_pts.flatten(0, 2), encode_theta=True).reshape(D, W, H, -1) # D, W, H, embed_dim
            local_pos_encode = self.pos_encoder(sph_ref_pts.flatten(0, 2), encode_theta=False).reshape(D, W, H, -1)
            
            # ref_pts: Ncam, bs, D, W, H, 2 (x, y pixel coords)
            # per_cam_mask_list: Ncam, bs, D, W, H (boolean)
            ref_pts_3d, ref_pts, per_cam_mask_list, cam_pts = self.point_sampling(
                wrld_ref_3d, cam_params=cam_params)
            
            overlap = []

            for j in range(bs):
                batch_overlap = []
                for pair in self.cam_pairs:
                    cam1, cam2 = pair
                    mask1 = per_cam_mask_list[cam1, j]
                    mask2 = per_cam_mask_list[cam2, j]
                    per_batch = (mask1 * mask2).unsqueeze(0) # 1, D, W, H boolean
                    batch_overlap.append(per_batch)
                batch_overlap = torch.any(torch.cat(batch_overlap, dim=0), dim=0)
                overlap.append(batch_overlap)
                
            num_cams = ref_pts.shape[0]
            max_len = 0
            
            # indexes: [bs][Ncam][N][3] list
            # sph_ref_pts[index_query_per_img]: N, 3 (Spherical Coords)
            # Make positional encoding per cam: [bs][Ncam] list (N, pos_encode_embed_dim) size
            # Overlapped indexes: [bs][Noverlap][3]
            for j in range(bs):
                overlap_per_batch = overlap[j].nonzero()
                overlapped_indexes.append(overlap_per_batch)
                for i, per_cam_mask in enumerate(per_cam_mask_list):
                    index_query_per_img = per_cam_mask[j].nonzero().squeeze(-1)
                    if len(index_query_per_img) == 0:
                        index_query_per_img = per_cam_mask_list[i][j].nonzero().squeeze(-1)[0:1]
                    indexes[j].append(index_query_per_img)
                    ####################################################
                    # Positional Encoding per Cam.
                    local_pos = sph_ref_pts[index_query_per_img[:,0], index_query_per_img[:,1], index_query_per_img[:,2]]
                    local_encode = local_pos_encode[index_query_per_img[:,0], index_query_per_img[:,1], index_query_per_img[:,2]]
                    # max(local_pos[...,1])+min(local_pos[...,1])
                    if i == 4:
                        local_pos[local_pos[..., 1] < 0] += 2*torch.pi
                    local_pos[..., 1] -= self.cam_theta[i] * torch.pi /180
                    theta = self.pos_encoder(local_pos, theta_only = True)
                    local_encode = torch.cat([local_encode, theta], dim=-1)
                    pos_encode_indexes[j].append(local_encode)
                    ####################################################
                    max_len = max(max_len, len(index_query_per_img))

            # queries_rebatch = torch.zeros(
            #     [bs, num_cams, max_len, self.embed_dims]
            #     ).cuda()
            
            local_pos_per_cam = torch.zeros(
                [bs, num_cams, max_len, self.embed_dims]
                ).cuda()
            
            reference_points_rebatch = torch.zeros(
                [bs, num_cams, max_len, 2]
                ).cuda()
            
            # global_pos_per_cam = torch.zeros(
            #     [bs, num_cams, max_len, self.embed_dims]
            #     ).cuda()

            for j in range(bs):
                for i, reference_points_per_img in enumerate(ref_pts):   
                    index_query_per_img = indexes[j][i]
                    d_indices = index_query_per_img[:, 0]
                    w_indices = index_query_per_img[:, 1]
                    h_indices = index_query_per_img[:, 2]
                    # queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                    reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, d_indices, w_indices, h_indices]
                    local_pos_per_cam[j, i, :len(index_query_per_img)] = pos_encode_indexes[j][i][:len(index_query_per_img)]
                    # global_pos_per_cam[j, i, :len(index_query_per_img)] = global_pos_encode[d_indices, w_indices, h_indices]
            # query = queries_rebatch.flatten(start_dim=0, end_dim=1).float()
            query = local_pos_per_cam.flatten(start_dim=0, end_dim=1).float()
            ref_pts = reference_points_rebatch.flatten(start_dim=0, end_dim=1).unsqueeze(2).repeat(1, 1, self.num_levels, 1)

            # Value: Bs, Ncam, C, H, W => Bs*Ncam, H*W, C
            value = value.flatten(start_dim=0, end_dim=1).flatten(start_dim=2, end_dim=3).permute(0,2,1).float()

        if bev_value:
            ref_pts = ref_pts.unsqueeze(2).repeat(1, 1, self.num_levels, 1)[..., :2]

        if occ_value:
            ref_pts = ref_pts.unsqueeze(2).repeat(1, 1, self.num_levels, 1)

        query = self.attn(
            query,
            value=value,
            # query_pos=query_pos,
            reference_points=ref_pts,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index
            )
        
        if not hasattr(self, 'ffn'):
            return query
        query = query + self.ffn(self.norm2(query))

        if view_transform:
            query = query.reshape(bs, num_cams, -1, self.embed_dims)
            return query, indexes, overlapped_indexes, global_pos_encode
            # return query, indexes, overlapped_indexes, global_pos_per_cam, output_map, global_pos_encode
        
        else:
            return query
        
@NECKS.register_module()
class MLPGeometryHead(nn.Module):
    def __init__(self,
                 input_channels=None,
                 threshold=None):
        super(MLPGeometryHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=1, stride=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 64, kernel_size=1, stride=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        self.threshold = threshold

    def forward(self, x):
        
        bs, D, W, H, _ = x.shape
        x = x.flatten(1, 3).permute(0, 2, 1).contiguous()
        x = self.mlp(x).squeeze(-1)
        x = x.reshape(bs, D, W, H)

        result = (x > self.threshold)
        
        return x, result
    
class _ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()
    
    @force_fp32()
    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(
            inplanes,
            mid_channels,
            1,
            padding=0,
            dilation=dilations[0],
            BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[3],
            dilation=dilations[3],
            BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(
            int(mid_channels * 5), mid_channels, 1, bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()
    
    @force_fp32()
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(
            x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


@NECKS.register_module()
class cam_feat_encoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        self.aspp = ASPP(in_channel, out_channel)
    
    def forward(self, x):
        x = x.flatten(0, 1)
        x = self.aspp(x)

        return x


@NECKS.register_module()
class CamPosEncoder(nn.Module):
    def __init__(self, data_config = None, grid_config = None):
        super(CamPosEncoder, self).__init__()
        self.original_dim=data_config['input_size']
        self.cam_theta = grid_config['Cam_Setting']
        
    def img_to_spherical(self, img_grid, intrinsic, rotation, trans, post_rotation, post_trans, bda, cam_theta):
        """
        Args:
            u, v: Pixel coordinates in the image (bs, ncam, H, W)
            intrinsic: Camera intrinsic matrix (bs, ncam, 3, 3)
            rotation: Camera rotation matrix (bs, ncam, 3, 3)
            trans: Camera translation vector (bs, ncam, 3)
            post_rotation: Post-rotation matrix (bs, ncam, 3, 3)
            post_trans: Post-translation vector (bs, ncam, 3)
            bda: BDA rotation matrix (bs, 3, 3)

        Returns:
            theta: Azimuth angles (bs, ncam, H, W)
            phi: Elevation angles (bs, ncam, H, W)
        """
        bs, Ncam, N, _ = img_grid.shape

        ones = torch.ones((bs, Ncam, N), device=img_grid.device).unsqueeze(-1)  # Z = 1 for depth
        ones *= 1
        img_coords = torch.cat([img_grid, ones], dim=-1)  # Shape (bs, ncam, HW, 3)

        img_coords -= post_trans.view(bs, Ncam, 1, 3)
        img_coords = torch.inverse(post_rotation).view(bs, Ncam, 1, 3, 3).matmul(img_coords.unsqueeze(-1)).squeeze(-1)

        img_coords = intrinsic.inverse().view(bs, Ncam, 1, 3, 3).matmul(img_coords.unsqueeze(-1)).squeeze(-1)

        # img_coords = rotation.view(bs, Ncam, 1, 3, 3).matmul(img_coords.unsqueeze(-1)).squeeze(-1)

        # img_coords += trans.view(bs, Ncam, 1, 3)

        final_coords = bda.view(bs, 1, 1, 3, 3).matmul(img_coords.unsqueeze(-1)).squeeze(-1)

        W, H, D = final_coords[..., 0], final_coords[..., 1], final_coords[..., 2]
        print(W, H, D)
        theta = torch.atan2(W, D)  # Shape (bs, ncam, H*W)
        phi = torch.atan2(H, torch.sqrt(D**2 + W**2))  # Shape (bs, ncam, H*W)

        for i, theta_cam in enumerate(cam_theta):
            # theta[:, i] -= theta_cam
            print(min(theta[0][i]), max(theta[0][i]))
        print(phi)
        assert False
        return theta, phi

    def forward(self, img_context, rot, tran, intrin, post_rot, post_tran, bda):
        bs, Ncam, _, H, W = img_context.shape
        dwn_ratio = self.original_dim[0] // H
        assert (self.original_dim[0] // H) == (self.original_dim[1] // W)

        # intrin[..., 0, 0] /= dwn_ratio
        # intrin[..., 1, 1] /= dwn_ratio
        # intrin[..., 0, 2] /= dwn_ratio
        # intrin[..., 1, 2] /= dwn_ratio

        u = torch.linspace(0, W - 1, W).to(img_context.device)
        v = torch.linspace(0, H - 1, H).to(img_context.device)

        u, v = torch.meshgrid(u, v, indexing='ij')
        img_grid = torch.stack([u, v], dim=-1).flatten(0, 1) * dwn_ratio
        # print(rot, post_rot)
        # print(intrin)
        # print(tran, bda, post_tran)
 
        img_grid = img_grid[None, None, :].expand(bs, Ncam, H*W, 2)
        theta, phi = self.img_to_spherical(img_grid, intrin, rot, tran, post_rot, post_tran, bda, self.cam_theta)

        return theta