data_root = 'data/SSCBenchWaymo'
ann_file = 'data/SSCBenchWaymo/unified/labels'
stereo_depth_root = 'data/SSCBenchWaymo/depth'
camera_used = ['image_1']

gpu=2

dataset_type = 'WaymoSSCBenchDataset'
point_cloud_range = [0, -25.6, -2, 51.2, 25.6, 4.4]
occ_size = [256, 256, 32]

waymo_class_frequencies = [6609676234, 80263507, 150778, 14621, 8407737, 314538602, 79803104, 119425715, 180969842, 228187327, 8890485]

# 10 classes with unlabeled
class_names = [
    "unlabeled",
    "car",
    "bicycle",
    "motorcycle",
    "person",
    "road",
    "sidewalk",
    "other-ground",
    "building",
    "vegetation",
    "other-object",
]
num_class = len(class_names)

# dataset config #
bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
    flip_dz_ratio=0
)

data_config={
    'input_size': (1280, 1920),
    # 'resize': (-0.06, 0.11),
    # 'rot': (-5.4, 5.4),
    # 'flip': True,
    'resize': (0.5, 0.5),
    'rot': (0.0, 0.0 ),
    'flip': (0.0, 0.0 ),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', data_config=data_config, load_stereo_depth=True,
         is_train=True, color_jitter=(0.4, 0.4, 0.4)),
    dict(type='CreateDepthFromLiDAR', data_root=data_root, dataset='waymo', load_seg=False),
    dict(type='LoadAnnotationOcc', bda_aug_conf=bda_aug_conf, apply_bda=False,
            is_train=True, point_cloud_range=point_cloud_range),
    dict(type='CollectData', keys=['img_inputs', 'gt_occ'], 
            meta_keys=['pc_range', 'occ_size', 'raw_img', 'stereo_depth', 'focal_length', 'img_shape', 'gt_depths']),
]

trainset_config=dict(
    type=dataset_type,
    stereo_depth_root=stereo_depth_root,
    data_root=data_root,
    ann_file=ann_file,
    pipeline=train_pipeline,
    split='train',
    camera_used=camera_used,
    occ_size=occ_size,
    pc_range=point_cloud_range,
    test_mode=False,
)

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', data_config=data_config, load_stereo_depth=True,
         is_train=False, color_jitter=None),
    dict(type='CreateDepthFromLiDAR', data_root=data_root, dataset='waymo'),
    dict(type='LoadAnnotationOcc', bda_aug_conf=bda_aug_conf, apply_bda=False,
            is_train=False, point_cloud_range=point_cloud_range),
    dict(type='CollectData', keys=['img_inputs', 'gt_occ'],  
            meta_keys=['pc_range', 'occ_size', 'sequence', 'frame_id', 'raw_img', 'stereo_depth', 'focal_length', 'img_shape', 'gt_depths'])
]

testset_config=dict(
    type=dataset_type,
    stereo_depth_root=stereo_depth_root,
    data_root=data_root,
    ann_file=ann_file,
    pipeline=test_pipeline,
    split='test',
    camera_used=camera_used,
    occ_size=occ_size,
    pc_range=point_cloud_range
)

data = dict(
    train=trainset_config,
    val=testset_config,
    test=testset_config
)

train_dataloader_config = dict(
    batch_size=1,
    num_workers=4)

test_dataloader_config = dict(
    batch_size=1,
    num_workers=4)

# model
numC_Trans = 128
lss_downsample = [2, 2, 2]
voxel_out_channels = [128]
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]

grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_x * lss_downsample[0]],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_y * lss_downsample[1]],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_z * lss_downsample[2]],
    'dbound': [2.0, 58.0, 0.5],
}

_num_layers_cross_ = 3
_num_points_cross_ = 8
_num_levels_ = 1
_num_cams_ = 1
_dim_ = 128
_pos_dim_ = _dim_//2

_num_layers_self_ = 2
_num_points_self_ = 8

model = dict(
    type='SimpleLSSFormer',
    img_backbone=dict(
        type='CustomEfficientNet',
        arch='b4',
        drop_path_rate=0.2,
        frozen_stages=0,
        norm_eval=False,
        out_indices=(2, 3, 4, 5, 6),
        with_cp=True,
        init_cfg=dict(type='Pretrained', prefix='backbone', 
        checkpoint='./ckpts/efficientnet-b4_3rdparty_8xb32-aa_in1k_20220119-45b8bd2b'),
    ),
    img_neck=dict(
        type='SECONDFPN',
        in_channels= [32, 56, 160, 448, 1792], # [48, 80, 224, 640, 2560],
        upsample_strides=[0.5, 1, 2, 4, 4], 
        out_channels=[128, 128, 128, 128, 128]),
    depth_net=dict(
        type='GeometryDepth_Net',
        downsample=8,
        numC_input=640,
        numC_Trans=numC_Trans,
        cam_channels=33,
        grid_config=grid_config,
        loss_depth_type='kld',
        loss_depth_weight=0.0001,
    ),
    img_view_transformer=dict(
        type='LSSViewTransformer',
        downsample=8,
        grid_config=grid_config,
        data_config=data_config,
    ),
    proposal_layer=dict(
        type='VoxelProposalLayer',
        point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
        input_dimensions=[128, 128, 16],
        data_config=data_config,
        init_cfg=None
    ),
    VoxFormer_head=dict(
        type='VoxFormerHead',
        volume_h=128,
        volume_w=128,
        volume_z=16,
        data_config=data_config,
        point_cloud_range=point_cloud_range,
        embed_dims=_dim_,
        cross_transformer=dict(
           type='PerceptionTransformer_DFA3D',
           rotate_prev_bev=True,
           use_shift=True,
           embed_dims=_dim_,
           num_cams = _num_cams_,
           encoder=dict(
               type='VoxFormerEncoder_DFA3D',
               num_layers=_num_layers_cross_,
               pc_range=point_cloud_range,
               data_config=data_config,
               num_points_in_pillar=8,
               return_intermediate=False,
               transformerlayers=dict(
                   type='VoxFormerLayer',
                   attn_cfgs=[
                       dict(
                           type='DeformCrossAttention_DFA3D',
                           pc_range=point_cloud_range,
                           num_cams=_num_cams_,
                           deformable_attention=dict(
                               type='MSDeformableAttention3D_DFA3D',
                               embed_dims=_dim_,
                               num_points=_num_points_cross_,
                               num_levels=_num_levels_),
                           embed_dims=_dim_,
                       )
                   ],
                   ffn_cfgs=dict(
                       type='FFN',
                       embed_dims=_dim_,
                       feedforward_channels=1024,
                       num_fcs=2,
                       ffn_drop=0.,
                       act_cfg=dict(type='ReLU', inplace=True),
                   ),
                   feedforward_channels=_dim_ * 2,
                   ffn_dropout=0.1,
                   operation_order=('cross_attn', 'norm', 'ffn', 'norm')))),
        self_transformer=dict(
           type='PerceptionTransformer_DFA3D',
           rotate_prev_bev=True,
           use_shift=True,
           embed_dims=_dim_,
           num_cams = _num_cams_,
           use_level_embeds = False,
           use_cams_embeds = False,
           encoder=dict(
               type='VoxFormerEncoder',
               num_layers=_num_layers_self_,
               pc_range=point_cloud_range,
               data_config=data_config,
               num_points_in_pillar=8,
               return_intermediate=False,
               transformerlayers=dict(
                   type='VoxFormerLayer',
                   attn_cfgs=[
                       dict(
                           type='DeformSelfAttention',
                           embed_dims=_dim_,
                           num_levels=1,
                           num_points=_num_points_self_)
                   ],
                   ffn_cfgs=dict(
                       type='FFN',
                       embed_dims=_dim_,
                       feedforward_channels=1024,
                       num_fcs=2,
                       ffn_drop=0.,
                       act_cfg=dict(type='ReLU', inplace=True),
                   ),
                   feedforward_channels=_dim_ * 2,
                   ffn_dropout=0.1,
                   operation_order=('self_attn', 'norm', 'ffn', 'norm')))),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=512,
            col_num_embed=512,
           ),
        mlp_prior=True
    ),
    occ_encoder_backbone=dict(
        type='LocalAggregator',
        local_encoder_backbone=dict(
            type='CustomResNet3D',
            numC_input=128,
            num_layer=[2, 2, 2],
            num_channels=[128, 128, 128],
            stride=[1, 2, 2]
        ),
        local_encoder_neck=dict(
            type='GeneralizedLSSFPN',
            in_channels=[128, 128, 128],
            out_channels=_dim_,
            start_level=0,
            num_outs=3,
            norm_cfg=norm_cfg,
            conv_cfg=dict(type='Conv3d'),
            act_cfg=dict(
                type='ReLU',
                inplace=True),
            upsample_cfg=dict(
                mode='trilinear',
                align_corners=False
            )
        )
    ),
    pts_bbox_head=dict(
        type='OccHead',
        in_channels=[sum(voxel_out_channels)],
        out_channel=num_class,
        empty_idx=0,
        num_level=1,
        with_cp=True,
        occ_size=occ_size,
        loss_weight_cfg = {
                "loss_voxel_ce_weight": 1.0,
                "loss_voxel_sem_scal_weight": 1.0,
                "loss_voxel_geo_scal_weight": 1.0
        },
        conv_cfg=dict(type='Conv3d', bias=False),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        class_frequencies=waymo_class_frequencies
    )
)

"""Training params."""
learning_rate=3e-4
training_steps=54000

optimizer = dict(
    type="AdamW",
    lr=learning_rate,
    weight_decay=0.01
)

lr_scheduler = dict(
    type="OneCycleLR",
    max_lr=learning_rate,
    total_steps=training_steps + 10,
    pct_start=0.05,
    cycle_momentum=False,
    anneal_strategy="cos",
    interval="step",
    frequency=1
)