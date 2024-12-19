import torch
import torch.nn as nn
from mmcv.runner import BaseModule



class FLoSP(BaseModule):
    def __init__(self, scene_size=(256,256,32), dataset=None, project_scale=2, project_res=[]):
        super().__init__()
        self.scene_size = scene_size
        self.dataset = dataset
        self.project_scale = project_scale
        self.project_res = project_res

    def forward_single(self, x2d, projected_pix, fov_mask):
        c, h, w = x2d.shape

        src = x2d.view(c, -1)
        zeros_vec = torch.zeros(c, 1).type_as(src)
        src = torch.cat([src, zeros_vec], 1)

        pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]
        img_indices = pix_y * w + pix_x
        img_indices[~fov_mask] = h * w
        img_indices = img_indices.expand(c, -1).long()  # c, HWD
        src_feature = torch.gather(src, 1, img_indices)
    
        x3d = src_feature.reshape(
            c,
            self.scene_size[0] // self.project_scale,
            self.scene_size[1] // self.project_scale,
            self.scene_size[2] // self.project_scale,
        )

        return x3d
    
    def forward(self, feat, cam_params, img_metas):
        B, N, C, H, W = feat.shape
        rots, trans, intrins, post_rots, post_trans, bda = cam_params
        
        x3ds = []
        for i in range(B):
            x3d = None
            for scale_2d in self.project_res:

                # project features at each 2D scale to target 3D scale
                scale_2d = int(scale_2d)
                projected_pix = img_metas["projected_pix_{}".format(self.project_scale)][i].cuda()
                fov_mask = img_metas["fov_mask_{}".format(self.project_scale)][i].cuda()

                # Sum all the 3D features
                if x3d is None:
                    x3d = self.projects[str(scale_2d)](
                        feat["1_" + str(scale_2d)][i],
                        projected_pix // scale_2d,
                        fov_mask,
                    )
                else:
                    x3d += self.projects[str(scale_2d)](
                        feat["1_" + str(scale_2d)][i],
                        projected_pix // scale_2d,
                        fov_mask,
                    )
            x3ds.append(x3d)

        x3d = torch.stack(x3ds)
        
        return x3d