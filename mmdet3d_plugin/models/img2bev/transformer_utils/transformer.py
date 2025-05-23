# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/VoxFormer/blob/main/LICENSE

# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import normal_
from torchvision.transforms.functional import rotate
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule
from .deformable_self_attention import DeformSelfAttention
from .deformable_cross_attention import MSDeformableAttention3D
from mmcv.ops import MultiScaleDeformableAttention


@TRANSFORMER.register_module()
class PerceptionTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 encoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_cams_embeds=True,
                 use_level_embeds=True,
                 rotate_center=[100, 100],
                 **kwargs):
        super(PerceptionTransformer, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_cams_embeds = use_cams_embeds
        self.use_level_embeds = use_level_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        if self.use_level_embeds:
            self.level_embeds = nn.Parameter(torch.Tensor(
                self.num_feature_levels, self.embed_dims))
        
        if self.use_cams_embeds:
            self.cams_embeds = nn.Parameter(
                torch.Tensor(self.num_cams, self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, DeformSelfAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def get_vox_features(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            ref_3d,
            vox_coords,
            unmasked_idx,
            cam_params=None,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        """
        obtain voxel features.
        """

        bs = mlvl_feats[0].size(0)
        # To do, implement a function which supports bs > 1
        assert bs == 1
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1) #  #[N, 1, 64]
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1) # [N, 1, 64]

        unmasked_bev_queries = bev_queries[vox_coords[unmasked_idx, 3], :, :]
        unmasked_bev_bev_pos = bev_pos[vox_coords[unmasked_idx, 3], :, :]

        unmasked_ref_3d = ref_3d[vox_coords[unmasked_idx, 3], :]
        unmasked_ref_3d = unmasked_ref_3d.unsqueeze(0).unsqueeze(0).to(unmasked_bev_queries.device)
        
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)

        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=bev_pos.device)

        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        bev_embed = self.encoder(
            unmasked_bev_queries,
            feat_flatten,
            feat_flatten,
            ref_3d=unmasked_ref_3d,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=unmasked_bev_bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            cam_params=cam_params,
            prev_bev=None,
            shift=None,
            **kwargs
        )

        return bev_embed

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def diffuse_vox_features(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            ref_3d,
            vox_coords,
            unmasked_idx,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        """
        diffuse voxel features.
        """

        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1) 
        if bev_pos is not None:
            bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        unmasked_ref_3d = ref_3d[vox_coords[unmasked_idx, 3], :]
        unmasked_ref_3d = unmasked_ref_3d.unsqueeze(0).unsqueeze(0).to(bev_queries.device)
        
        bev_embed = self.encoder(
            bev_queries,
            None,
            None,
            ref_3d=unmasked_ref_3d,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=None,
            level_start_index=None,
            prev_bev=None,
            shift=None,
            **kwargs
        ) 
        
        return bev_embed

@TRANSFORMER.register_module()
class PerceptionTransformer_DFA3D(PerceptionTransformer):
    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def get_vox_features(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            ref_3d,
            vox_coords,
            unmasked_idx,
            cam_params=None,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            mlvl_dpt_dists=None,
            **kwargs):
        """
        obtain voxel features.
        """

        bs = mlvl_feats[0].size(0)
        # To do, implement a function which supports bs > 1
        assert bs == 1
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1) #  #[N, 1, 64]
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1) # [N, 1, 64]

        unmasked_bev_queries = bev_queries[vox_coords[unmasked_idx, 3], :, :]
        unmasked_bev_bev_pos = bev_pos[vox_coords[unmasked_idx, 3], :, :]

        unmasked_ref_3d = ref_3d[vox_coords[unmasked_idx, 3], :]
        unmasked_ref_3d = unmasked_ref_3d.unsqueeze(0).unsqueeze(0).to(unmasked_bev_queries.device)
        
        feat_flatten = []
        spatial_shapes = []
        dpt_dist_flatten = []
        for lvl, (feat, dpt_dist) in enumerate(zip(mlvl_feats, mlvl_dpt_dists)):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            dpt_dist = dpt_dist.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
            dpt_dist_flatten.append(dpt_dist)

        feat_flatten = torch.cat(feat_flatten, 2)
        dpt_dist_flatten = torch.cat(dpt_dist_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        bev_embed = self.encoder(
            unmasked_bev_queries,
            feat_flatten,
            feat_flatten,
            value_dpt_dist=dpt_dist_flatten,
            ref_3d=unmasked_ref_3d,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=unmasked_bev_bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            cam_params=cam_params,
            prev_bev=None,
            shift=None,
            **kwargs
        )

        return bev_embed
    
    
class TransformerLayer(nn.Module):

    def __init__(self, embed_dims, num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.embed_dims = embed_dims
        self.norm1 = norm_layer(embed_dims)
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, bias=qkv_bias, batch_first=True)

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


class DeformableTransformerLayer(nn.Module):

    def __init__(self,
                 embed_dims,
                 num_heads=8,
                 num_levels=3,
                 num_points=4,
                 mlp_ratio=4,
                 attn_layer=MultiScaleDeformableAttention,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()
        self.embed_dims = embed_dims
        self.norm1 = norm_layer(embed_dims)
        self.attn = attn_layer(
            embed_dims, num_heads, num_levels, num_points, batch_first=True, **kwargs)

        if mlp_ratio == 0:
            return
        self.norm2 = norm_layer(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dims * mlp_ratio, embed_dims),
        )

    def forward(self,
                query,
                value=None,
                query_pos=None,
                ref_pts=None,
                spatial_shapes=None,
                level_start_index=None):
        query = query + self.attn(
            self.norm1(query),
            value=value,
            query_pos=query_pos,
            reference_points=ref_pts,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index)
        if not hasattr(self, 'ffn'):
            return query
        query = query + self.ffn(self.norm2(query))
        return query