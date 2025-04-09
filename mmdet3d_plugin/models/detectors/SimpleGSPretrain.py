import torch
from mmcv.cnn import Conv2d
from mmcv.runner import BaseModule,force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.models import builder
from mmdet3d_plugin.models.utils.uni3d_voxelpooldepth import DepthNet
import pickle
import numpy as np
from pdb import set_trace

@DETECTORS.register_module()
class SimpleGSPretrain(BaseModule):
    def __init__(
        self,
        img_backbone,
        img_neck,
        depth_head=None,
        pts_bbox_head=None,
        train_cfg=None,
        test_cfg=None
    ):
        super().__init__()

        self.img_backbone = builder.build_backbone(img_backbone)
        self.img_neck = builder.build_neck(img_neck)
        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = builder.build_head(pts_bbox_head)
        
        if self.with_img_backbone:
            in_channels = self.img_neck.out_channels
            out_channels = self.pts_bbox_head.in_channels
            if isinstance(in_channels, list):
                in_channels = in_channels[0]
            self.input_proj = Conv2d(in_channels, out_channels, kernel_size=1)
        
            if depth_head is not None:
                    depth_dim = self.pts_bbox_head.view_trans.depth_dim
                    dhead_type = depth_head.pop("type", "SimpleDepth")
                    if dhead_type == "SimpleDepth":
                        self.depth_net = Conv2d(out_channels, depth_dim, kernel_size=1)
                    else:
                        self.depth_net = DepthNet(
                            out_channels, out_channels, depth_dim, **depth_head
                        )
            self.depth_head = depth_head
        
    @property
    def with_img_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, 'img_backbone') and self.img_backbone is not None
        
    @property
    def with_depth_head(self):
        """bool: Whether the detector has a depth head."""
        return hasattr(self, "depth_head") and self.depth_head is not None

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
    
    @auto_fp16(apply_to=("img"))
    def pred_depth(self, img, img_metas, img_feats=None):
        if img_feats is None or not self.with_depth_head:
            return None
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        depth = []
        
        # set_trace() # img_feats - torch.Size([1, 1, 512, 48, 176])
        
        for _feat in img_feats:
            _depth = self.depth_net(_feat.view(-1, *_feat.shape[-3:]))
            _depth = _depth.softmax(dim=1)
            depth.append(_depth)
        return depth
    
    def extract_img_feat(self, img_inputs, img_metas):
        img_enc_feats = self.image_encoder(img_inputs[0])
        depth = self.pred_depth(img=img_inputs[0], img_metas=img_metas, img_feats=img_enc_feats)
        
        pts_feats = None

        return pts_feats, img_enc_feats, depth

    @force_fp32(apply_to=("pts_feats", "img_feats"))
    def forward_pts_train(
        self, pts_feats, img_feats, points, img_inputs, img_metas, img_depth
    ):
        batch_rays = self.pts_bbox_head.sample_rays(points, img_inputs, img_metas)

        out_dict = self.pts_bbox_head(
            pts_feats, img_feats, img_metas, img_depth, batch_rays
        )
        losses = self.pts_bbox_head.loss(out_dict, batch_rays)

        if self.with_depth_head and hasattr(self.pts_bbox_head.view_trans, "loss"):
            losses.update(
                self.pts_bbox_head.view_trans.loss(img_depth, points, img_inputs[0], img_metas)
            )

        return losses
    
    def forward_train(self, data_dict):
        img_inputs = data_dict['img_inputs']
        img_metas = data_dict['img_metas']
        points = data_dict['points']
        
        pts_feats, img_feats, img_depth = self.extract_img_feat(img_inputs, img_metas)
        
        losses = dict()
        
        losses_pts = self.forward_pts_train(
            pts_feats, img_feats, points, img_inputs, img_metas, img_depth
        )
        losses.update(losses_pts)
        
        train_output = { 'losses': losses }

        return train_output
    
    def forward_test(self, data_dict):
        img_inputs = data_dict['img_inputs']
        img_metas = data_dict['img_metas']
        points = data_dict['points']
        
        results = self.simple_test(img_metas=img_metas, points=points, img_inputs=img_inputs)

        return results
    
    def simple_test(self, img_metas, points=None, img_inputs=None):
        """Test function without augmentaiton."""
        pts_feats, img_feats, img_depth = self.extract_img_feat(img_inputs=img_inputs, img_metas=img_metas)
        batch_rays = self.pts_bbox_head.sample_rays_test(points, img_inputs, img_metas)
        results = self.pts_bbox_head(
            pts_feats=pts_feats,  img_feats=img_feats, img_metas=img_metas, img_depth=img_depth, batch_rays=batch_rays
        )
        # with open("outputs/{}.pkl".format(img_metas[0]["sample_idx"]), "wb") as f:
        #     H, W = img_metas[0]["img_shape"][0][0], img_metas[0]["img_shape"][0][1]
        #     num_cam = len(img_metas[0]["img_shape"])
        #     l = 2
        #     init_weights = results[0]["vis_weights"]
        #     init_weights = init_weights.reshape(num_cam, -1, *init_weights.shape[1:])
        #     init_sampled_points = results[0]["vis_sampled_points"]
        #     init_sampled_points = init_sampled_points.reshape(
        #         num_cam, -1, *init_sampled_points.shape[1:]
        #     )
        #     pts_idx = np.random.randint(
        #         0, high=init_sampled_points.shape[1], size=(256,), dtype=int
        #     )
        #     init_weights = init_weights[:, pts_idx]
        #     init_sampled_points = init_sampled_points[:, pts_idx]
        #     pickle.dump(
        #         {
        #             "render_rgb": results[0]["rgb"]
        #             .reshape(num_cam, H // l, W // l, 3)
        #             .detach()
        #             .cpu()
        #             .numpy(),
        #             "render_depth": results[0]["depth"]
        #             .reshape(num_cam, H // l, W // l, 1)
        #             .detach()
        #             .cpu()
        #             .numpy(),
        #             "rgb": batch_rays[0]["rgb"].detach().cpu().numpy(),
        #             "scaled_points": results[0]["scaled_points"].detach().cpu().numpy(),
        #             "points": points[0].detach().cpu().numpy(),
        #             "lidar2img": np.asarray(img_metas[0]["lidar2img"])[
        #                 :, 0
        #             ],  # (N, 4, 4)
        #             # 'weights': results[0]['vis_weights'].detach().cpu().numpy(),
        #             # 'sampled_points': results[0]['vis_sampled_points'].detach().cpu().numpy(),
        #             "init_weights": init_weights.detach().cpu().numpy(),
        #             "init_sampled_points": init_sampled_points.detach().cpu().numpy(),
        #         },
        #         f,
        #     )
        #     print("save to outputs/{}.pkl".format(img_metas[0]["sample_idx"]))
        return results

    def forward(self, data_dict):
        if self.training:
            return self.forward_train(data_dict)
        else:
            return self.forward_test(data_dict)