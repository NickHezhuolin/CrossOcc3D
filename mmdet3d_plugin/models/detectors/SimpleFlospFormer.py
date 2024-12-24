import torch
from mmcv.runner import BaseModule
from mmdet.models import DETECTORS
from mmdet3d.models import builder
from ..img2bev.Flosp import FLoSP
import torch.nn as nn
import time

@DETECTORS.register_module()
class SimpleFlospFormer(BaseModule):
    def __init__(
        self,
        img_backbone,
        img_neck,
        occ_encoder_backbone,
        view_scales,
        scene_size=(256, 256, 32), 
        volume_scale=2,
        pts_bbox_head=None,
        train_cfg=None,
        test_cfg=None
    ):
        super().__init__()

        self.img_backbone = builder.build_backbone(img_backbone)
        self.img_neck = builder.build_neck(img_neck)
        self.scene_size = scene_size
        self.volume_scale = volume_scale
        self.view_scales = view_scales
        self.project = FLoSP(scene_size, volume_scale)
        if occ_encoder_backbone is not None:
            self.occ_encoder_backbone = builder.build_backbone(occ_encoder_backbone)
        self.pts_bbox_head = builder.build_head(pts_bbox_head)
        
        self.reduce_conv = nn.Conv3d(640, 128, kernel_size=1, stride=1, padding=0)

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape   
        imgs = imgs.view(B * N, C, imH, imW)

        x = self.img_backbone(imgs)

        if self.img_neck is not None:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]

        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        
        return x
    
    def extract_img_feat(self, img_inputs, img_metas):
        img_enc_feats = self.image_encoder(img_inputs[0]) # B, N, C, H, W

        start_time = time.time() 
        projected_pix = img_metas[f'projected_pix']
        fov_mask = img_metas[f'fov_mask']
        coarse_queries = self.project(img_enc_feats, projected_pix, fov_mask)
        end_time = time.time()  # 结束计时
        print(f"Project Time taken: {end_time - start_time:.6f} seconds")

        return coarse_queries
    
    def occ_encoder(self, x):
        if hasattr(self, 'occ_encoder_backbone'):
            x = self.occ_encoder_backbone(x)
        
        if hasattr(self, 'occ_encoder_neck'):
            x = self.occ_encoder_neck(x)
        
        return x

    def forward_train(self, data_dict):
        img_inputs = data_dict['img_inputs']
        img_metas = data_dict['img_metas']
        gt_occ = data_dict['gt_occ']

        start_time = time.time() 
        img_voxel_feats = self.extract_img_feat(img_inputs, img_metas) # torch.Size([1, 640, 128, 128, 16]) BNCHWD
        end_time = time.time()  # 结束计时
        print(f"Img Feats Time taken: {end_time - start_time:.6f} seconds")

        conv_start_time = time.time() 
        # 输入: torch.Size([1, 640, 128, 128, 16]) -> torch.Size([1, 128, 128, 128, 16]) 
        img_voxel_feats_reduced = self.reduce_conv(img_voxel_feats.permute(0, 1, 4, 2, 3))  # 调整维度顺序以适配 Conv3d
        img_voxel_feats_reduced = img_voxel_feats_reduced.permute(0, 1, 3, 4, 2)  # 调整回原顺序
        voxel_feats_enc = self.occ_encoder(img_voxel_feats_reduced)
        conv_end_time = time.time()  # 结束计时
        print(f"Conv Time taken: {conv_end_time - conv_start_time:.6f} seconds")
        
        if len(voxel_feats_enc) > 1:
            voxel_feats_enc = [voxel_feats_enc[0]]
        
        if type(voxel_feats_enc) is not list:
            voxel_feats_enc = [voxel_feats_enc]

        output = self.pts_bbox_head(
            voxel_feats=voxel_feats_enc,
            img_metas=img_metas,
            img_feats=None,
            gt_occ=gt_occ
        )

        losses = dict()

        losses_occupancy = self.pts_bbox_head.loss(
            output_voxels=output['output_voxels'],
            target_voxels=gt_occ,
        )
        losses.update(losses_occupancy)

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        train_output = {
            'losses': losses,
            'pred': pred,
            'gt_occ': gt_occ
        }

        return train_output
    
    def forward_test(self, data_dict):
        img_inputs = data_dict['img_inputs']
        img_metas = data_dict['img_metas']
        gt_occ = data_dict['gt_occ']

        img_voxel_feats = self.extract_img_feat(img_inputs, img_metas)
        
        projected_pix = img_metas[f'projected_pix']
        fov_mask = img_metas[f'fov_mask']
        img_voxel_feats =  self.project(img_voxel_feats, projected_pix , fov_mask)
        
        voxel_feats_enc = self.occ_encoder(img_voxel_feats)

        if len(voxel_feats_enc) > 1:
            voxel_feats_enc = [voxel_feats_enc[0]]
        
        if type(voxel_feats_enc) is not list:
            voxel_feats_enc = [voxel_feats_enc]
        
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats_enc,
            img_metas=img_metas,
            img_feats=None,
            gt_occ=gt_occ
        )

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        test_output = {
            'pred': pred,
            'gt_occ': gt_occ
        }

        return test_output

    def forward(self, data_dict):
        if self.training:
            return self.forward_train(data_dict)
        else:
            return self.forward_test(data_dict)