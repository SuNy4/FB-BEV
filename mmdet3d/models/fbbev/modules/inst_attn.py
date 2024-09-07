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

@NECKS.register_module()
class TransformerLayer(nn.Module):

    def __init__(self, embed_dims, num_heads, mlp_ratio=4, kdim=None, vdim=None
                 , qkv_bias=True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.embed_dims = embed_dims
        self.norm1 = norm_layer(embed_dims)
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, bias=qkv_bias, 
                                          kdim=kdim, vdim=vdim, batch_first=True)

        if mlp_ratio == 0:
            return
        self.norm2 = norm_layer(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dims * mlp_ratio, embed_dims),
        )

    def forward(self, query, key=None, value=None, query_pos=None, key_pos=None):
        if key is None and value is None:
            key = value = query
            key_pos = query_pos
        if key_pos is not None:
            key = key + key_pos
        if query_pos is not None:
            query = query + self.attn(self.norm1(query) + query_pos, key, value)[0]
        else:
            query = query + self.attn(self.norm1(query), key, value)[0]
        if not hasattr(self, 'ffn'):
            return query
        query = query + self.ffn(self.norm2(query))
        return query


@NECKS.register_module()
class DeformableTransformerLayer(nn.Module):

    def __init__(self,
                 embed_dims,
                 num_heads=8,
                 num_levels=3,
                 num_points=4,
                 mlp_ratio=4,
                 grid_config=None,
                 data_config=None,
                 attn_layer=MultiScaleDeformableAttention,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()
        self.num_levels=num_levels
        self.x_bound = grid_config['x']
        self.y_bound = grid_config['y']
        self.z_bound = grid_config['z']
        self.embed_dims = embed_dims
        self.norm1 = norm_layer(embed_dims)
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
        self.original_dim=data_config['src_size']
        self.norm2 = norm_layer(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dims * mlp_ratio, embed_dims),
        )
        
    def generate_grid(self, grid_shape, value=None, offset=0, normalize=False):
        """
        Args:
            grid_shape: The (scaled) shape of grid.
            value: The (unscaled) value the grid represents.
        Returns:
            Grid coordinates of shape [len(grid_shape), *grid_shape]
        """
        if value is None:
            value = grid_shape
        grid = []
        for i, (s, val) in enumerate(zip(grid_shape, value)):
            g = torch.linspace(offset, val - 1 + offset, s, dtype=torch.float)
            if normalize:
                g /= s - 1
            shape_ = [1 for _ in grid_shape]
            shape_[i] = s
            g = g.reshape(1, *shape_).expand(1, *grid_shape)
            grid.append(g)
        return torch.cat(grid, dim=0)

    def get_reference_points(self, coords, dim='3d', device='cuda', dtype=torch.float):
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

            coords[..., 0] = (coords[..., 0] + self.x_bound[0]) / self.x_bound[-1]
            coords[..., 1] = (coords[..., 0] + self.y_bound[0]) / self.y_bound[-1]
            coords[..., 2] = (coords[..., 0] + self.z_bound[0]) / self.z_bound[-1]

            return coords
    
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
        B, N, H, W, D, _ = reference_points_cam.shape
        reference_points_cam = reference_points_cam.permute(1, 0, 2, 3, 4, 5).reshape(N, B, H*W, D, 3)
        mask = mask.permute(1, 0, 2, 3, 4, 5).reshape(N, B, H*W, D, 1).squeeze(-1)

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
                query,
                value=None,
                query_pos=None,
                ref_pts=None,
                img_value=False,
                bev_value=False,
                occ_value=False,
                spatial_shapes=None,
                level_start_index=None,
                cam_params=None,
                occ_size=None,
                ):

        bs, N, embed_dim = query.shape

        if img_value:

            indexes = [[] for _ in range(bs)]

            wrld_ref_3d = self.get_reference_points(
                coords=ref_pts,
                dim='2d', 
                device='cuda', 
                dtype=torch.float
                )
            
            # voxel_ref_3d = self.get_reference_points(
            #     coords=ref_pts,
            #     dim='3d', 
            #     device='cuda', 
            #     dtype=torch.float
            #     )
            
            ref_pts_3d, ref_pts, per_cam_mask_list, cam_pts = self.queries_point_sampling(
                wrld_ref_3d, cam_params=cam_params)
            
            # for i in range(6):
            #     print(per_cam_mask_list.shape)
            #     save = per_cam_mask_list[i][0].flatten(start_dim=0, end_dim=1).cpu()
                
            #     torch.save(save, f'meshgrid{i}.txt')
            # assert False
            # if bev_mask is not None:
            #     per_cam_mask_list_ = per_cam_mask_list & bev_mask[None, :, :, None]
            # else:
            #     per_cam_mask_list_ = per_cam_mask_list

            num_cams = ref_pts.shape[0]
            max_len = 0

            for j in range(bs):
                for i, per_cam_mask in enumerate(per_cam_mask_list):
                    index_query_per_img = per_cam_mask[j].nonzero().squeeze(-1)#sum(-1).nonzero().squeeze(-1)
                    if len(index_query_per_img) == 0:
                        index_query_per_img = per_cam_mask_list[i][j].nonzero().squeeze(-1)[0:1]#.sum(-1).nonzero().squeeze(-1)[0:1]
                    indexes[j].append(index_query_per_img)
                    max_len = max(max_len, len(index_query_per_img))

            queries_rebatch = query.new_zeros(
                [bs, num_cams, max_len, self.embed_dims]
                )
            # query_pos_rebatch = query_pos.new_zeros(
            #     [bs, num_cams, max_len, self.embed_dims]
            #     )
            reference_points_rebatch = ref_pts.new_zeros(
                [bs, num_cams, max_len, 2])
            
            for j in range(bs):
                for i, reference_points_per_img in enumerate(ref_pts):   
                    index_query_per_img = indexes[j][i]
                    queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                    #query_pos_rebatch[j, i, :len(index_query_per_img)] = query_pos[j, index_query_per_img]
                    reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]
            
            query = queries_rebatch.flatten(start_dim=0, end_dim=1).float()
            # query_pos = query_pos_rebatch.flatten(start_dim=0, end_dim=1).float()
            ref_pts = reference_points_rebatch.flatten(start_dim=0, end_dim=1).unsqueeze(2).repeat(1, 1, self.num_levels, 1)
            value = value.flatten(start_dim=0, end_dim=1).flatten(start_dim=2, end_dim=3).permute(0,2,1).float()

        if bev_value:
            ref_pts = ref_pts.unsqueeze(2).repeat(1, 1, self.num_levels, 1)[..., :2]

        if occ_value:
            ref_pts = ref_pts.unsqueeze(2).repeat(1, 1, self.num_levels, 1)
            
        query = query + self.attn(
            self.norm1(query),
            value=value,
            query_pos=query_pos,
            reference_points=ref_pts,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index
            )
        
        if not hasattr(self, 'ffn'):
            return query
        query = query + self.ffn(self.norm2(query))

        if img_value:
            output = torch.zeros([bs, N, embed_dim]).cuda()
            query = query.reshape(bs, num_cams, -1, embed_dim).permute(1,0,2,3)

            for j in range(bs):
                for i, query_per_cam in enumerate(query):
                    index_query_per_img = indexes[j][i]
                    output[j, index_query_per_img] = query_per_cam[j, :len(index_query_per_img)]

            return output
        
        else:
            return query
