data_root = 'data/SSCBenchKITTI360'
ann_file = 'data/SSCBenchKITTI360/unified/labels'
stereo_depth_root = 'data/SSCBenchKITTI360/depth'
camera_used = ['left']

gpu=4

dataset_type = 'KITTI360Dataset_half'
point_cloud_range = [0, -25.6, -2, 51.2, 25.6, 4.4]
occ_size = [256, 256, 32]

unified_voxel_size = [0.2, 0.2, 0.2]
frustum_range = [-1, -1, 2.0, -1, -1, 58.0]
frustum_size = [-1, -1, 1.0]

IMG_HEIGHT = 384
IMG_WIDTH = 1408

kitti360_class_frequencies =  [
    2305812911,
    123212463,
    96297,
    4051087,   
    45297267,
    110397082,  
    295883213,   
    50037503,    
    1561069,   
    30516166,
    1950115
]

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
    'input_size': (IMG_HEIGHT, IMG_WIDTH),
    'resize': (0., 0.),
    'rot': (0.0, 0.0 ),
    'flip': (0.0, 0.0 ),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

file_client_args = dict(backend="disk")

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', data_config=data_config, load_stereo_depth=True,
         is_train=False, color_jitter=None, znear=1.0, zfar=100.0),
    dict(type="LoadPointsFromVelo", data_root=data_root, dataset='kitti360', load_seg=False),
    dict(type='CreateDepthFromLiDAR', data_root=data_root, dataset='kitti360', load_seg=False),
    dict(type='LoadAnnotationOcc', bda_aug_conf=bda_aug_conf, apply_bda=False,
            is_train=True, point_cloud_range=point_cloud_range),
    dict(type='CollectData', keys=['img_inputs', 'gt_occ', 'points'], 
            meta_keys=['pc_range', 'occ_size', 'sequence', 'frame_id', 'raw_img', 
                       'stereo_depth', 'focal_length', 'baseline', 'img_shape', 
                       'gt_depths','lidar2img','cam_params', 'fovx', 'fovy', 'viewmatrix',
                       'projmatrix', 'cam_pos'
                       ])
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
         is_train=False, color_jitter=None, znear=1.0, zfar=100.0),
    dict(type="LoadPointsFromVelo", data_root=data_root, dataset='kitti360', load_seg=False),
    dict(type='CreateDepthFromLiDAR', data_root=data_root, dataset='kitti360'),
    dict(type='LoadAnnotationOcc', bda_aug_conf=bda_aug_conf, apply_bda=False,
            is_train=False, point_cloud_range=point_cloud_range),
    dict(type='CollectData', keys=['img_inputs', 'gt_occ', 'points'],  
            meta_keys=['pc_range', 'occ_size', 'sequence', 'frame_id', 'raw_img', 
                       'stereo_depth', 'focal_length', 'baseline', 'img_shape', 
                       'gt_depths','lidar2img','cam_params', 'fovx', 'fovy', 'viewmatrix',
                       'projmatrix', 'cam_pos'
                       ])
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
    num_workers=8)

# model
fp16_enabled = True
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
    type='SimpleGSPretrain',
    img_backbone=dict(
        type="MaskConvNeXt",
        arch="small",
        drop_path_rate=0.2,
        out_indices=(0, 1, 2, 3),
        norm_out=True,
        frozen_stages=1,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="/home/hez4sgh/1_work_dir/GaussianPretrain/data/ckpts/convnext_small_1k_224_ema.pth",
        ),
        mae_cfg=dict(
            downsample_scale=32, downsample_dim=768, mask_ratio=0.3, learnable=False
        ),
    ),
    img_neck=dict(
        type="FPN",
        in_channels=[96, 192, 384, 768],
        out_channels=128,
        start_level=1,
        add_extra_convs="on_output",
        num_outs=4,
        relu_before_extra_convs=True,
    ),
    depth_head=dict(type="SimpleDepth"),
    pts_bbox_head=dict(
        type="GaussianHead",
        fp16_enabled=False,
        in_channels=128,
        unified_voxel_size=unified_voxel_size,
        unified_voxel_shape=occ_size,
        pc_range=point_cloud_range,
        cam_nums=1,
        ray_sampler_cfg=dict(
            close_radius=3.0,
            far_radius=50.0,
            only_img_mask=False,
            only_point_mask=False,
            replace_sample=False,
            point_nsample=1024,
            point_ratio=0.99,
            pixel_interval=4,
            sky_region=0.4,
            merged_nsample=1024,
        ),
        view_cfg=dict(
            type="Uni3DViewTrans",
            frustum_range=frustum_range,
            frustum_size=frustum_size,
            num_convs=0,
            keep_sweep_dim=False,
            fp16_enabled=fp16_enabled,
        ),
         gs_cfg=dict(
            type="GSRegresser_Sample",
            voxel_size=unified_voxel_size,
            pc_range=point_cloud_range,
            voxel_shape=occ_size,
            max_scale=0.01,
            split_dimensions=[4, 3, 1, 3],
            interpolate_cfg=dict(type="SmoothSampler", padding_mode="zeros"),
            param_decoder_cfg=dict(
                    in_dim=32, out_dim=4+3+1+3, hidden_size=32, n_blocks=5
                ),
        ),
        img_H=IMG_HEIGHT,
        img_W=IMG_WIDTH,
        render_conv_cfg=dict(out_channels=32, kernel_size=3, padding=1),
        loss_cfg=dict(
            sensor_depth_truncation=0.1,
            sparse_points_sdf_supervised=False,
            weights=dict(
                depth_loss=1.0,
                rgb_loss=10.0,
                opacity_loss=0.0,
                opacity_focal_loss=10.0,
                lpips_loss=0.0,
                ssim_loss=0.0,
                occ_loss=0.0,
            ),
        ),
    ),
)


"""Training params."""
learning_rate=3e-4
training_steps=27000

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