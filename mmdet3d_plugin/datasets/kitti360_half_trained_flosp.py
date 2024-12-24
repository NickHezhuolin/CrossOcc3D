import os
import glob
import numpy as np
from numba import njit, prange
from mmdet.datasets import DATASETS
from torch.utils.data import Dataset
from mmdet.datasets.pipelines import Compose
import mmdet3d_plugin.utils.fusion as fusion

@DATASETS.register_module()
class KITTI360Dataset_half_flosp(Dataset):
    def __init__(
        self,
        data_root,
        label_root,
        stereo_depth_root,
        ann_file,
        pipeline,
        split,
        camera_used,
        occ_size,
        pc_range,
        project_scale=2,
        frustum_size=8,
        test_mode=False,
        load_continuous=False
    ):
        super().__init__()

        self.load_continuous = load_continuous
        self.splits = {
            "train": [
                "2013_05_28_drive_0000_sync", "2013_05_28_drive_0002_sync", "2013_05_28_drive_0003_sync", "2013_05_28_drive_0004_sync"
            ],
            "val": ["2013_05_28_drive_0006_sync"],
            "test": ["2013_05_28_drive_0009_sync"]
        }

        self.sequences = self.splits[split]

        self.data_root = data_root
        self.stereo_depth_root = stereo_depth_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.data_infos = self.load_annotations(self.ann_file)

        self.occ_size = occ_size
        self.pc_range = pc_range
        self.camera_map = {'left': '2', 'right': '3'}
        self.camera_used = [self.camera_map[camera] for camera in camera_used]
        
        self.frustum_size = frustum_size
        self.project_scale = project_scale
        self.output_scale = int(self.project_scale / 2)
        self.scene_size = (51.2, 51.2, 6.4)
        self.img_H = 376
        self.img_W = 1408
        
        self.label_root = label_root
        self.vox_origin = np.array([0, -25.6, -2])
        self.voxel_size = 0.2

        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        self._set_group_flag()
    
    def __len__(self):
        return len(self.data_infos)
    
    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            print('found None in training data')
            return None
        
        example = self.pipeline(input_dict)
        return example
    
    def prepare_test_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            print('found None in training data')
            return None
        
        example = self.pipeline(input_dict)
        return example
    
    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
    
    def get_data_info(self, index):
        info = self.data_infos[index]
        '''
        sample into includes the following:
            "img_2_path": img_2_path,
            "img_3_path": img_3_path,
            "sequence": sequence,
            "P2": P2,
            "P3": P3,
            "T_velo_2_cam": T_velo_2_cam,
            "proj_matrix_2": proj_matrix_2,
            "proj_matrix_3": proj_matrix_3,
            "voxel_path": voxel_path,
            "stereo_depth_path": stereo_depth_path
        '''
        input_dict = dict(
            occ_size = np.array(self.occ_size),
            pc_range = np.array(self.pc_range),
            sequence = info['sequence'],
            frame_id = info['frame_id'],
        )

        # load images, intrins, extrins, voxels
        image_paths = []
        lidar2cam_rts = []
        lidar2img_rts = []
        cam_intrinsics = []

        for cam_type in self.camera_used:
            image_paths.append(info['img_{}_path'.format(int(cam_type))])
            lidar2img_rts.append(info['proj_matrix_{}'.format(int(cam_type))])
            cam_intrinsics.append(info['P{}'.format(int(cam_type))])
            lidar2cam_rts.append(info['T_velo_2_cam'])
        
        focal_length = info['P2'][0, 0]
        baseline = self.dynamic_baseline(info)

        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                focal_length=focal_length,
                baseline=baseline
            ))
        input_dict['stereo_depth_path'] = info['stereo_depth_path']
        # gt_occ is None for test-set
        input_dict['gt_occ'] = self.get_ann_info(index, key='voxel_path')
        
        scale_3ds = [self.output_scale, self.project_scale]
        info["scale_3ds"] = scale_3ds
        cam_k = info['P2'][0:3, 0:3]
        info["cam_k"] = cam_k
        
        projected_pix_list = []
        fov_mask_list = []
        for scale_3d in scale_3ds:
            projected_pix, fov_mask, pix_z = vox2pix(
                info['T_velo_2_cam'],
                cam_k,
                self.vox_origin,
                self.voxel_size * scale_3d,
                self.img_W,
                self.img_H,
                self.scene_size,
            )
            info["projected_pix_{}".format(scale_3d)] = projected_pix
            info["fov_mask_{}".format(scale_3d)] = fov_mask
            info["pix_z_{}".format(scale_3d)] = pix_z
            
        target_1_path = os.path.join(self.label_root, info['sequence'], info['frame_id'] + "_1_1.npy")
        target = np.load(target_1_path)
        info["target"] = target
        target_8_path = os.path.join(self.label_root, info['sequence'], info['frame_id'] + "_1_8.npy")
        target_1_8 = np.load(target_8_path)
        CP_mega_matrix = compute_CP_mega_matrix(target_1_8)
        info["CP_mega_matrix"] = CP_mega_matrix

        # Compute the masks, each indicate the voxels of a local frustum
        if self.test_mode == False:
            projected_pix_output = info["projected_pix_{}".format(self.output_scale)]
            pix_z_output = info[
                "pix_z_{}".format(self.output_scale)
            ]
            frustums_masks, frustums_class_dists = compute_local_frustums(
                projected_pix_output,
                pix_z_output,
                target,
                self.img_W,
                self.img_H,
                dataset="kitti",
                n_classes=11,
                size=self.frustum_size,
            )
        else:
            frustums_masks = None
            frustums_class_dists = None
        info["frustums_masks"] = frustums_masks
        info["frustums_class_dists"] = frustums_class_dists
        
        input_dict.update(
            dict(
                projected_pix=info["projected_pix_{}".format(self.project_scale)],
                fov_mask=info["fov_mask_{}".format(self.project_scale)],
                frustums_masks=frustums_masks,
                frustums_class_dists=frustums_class_dists,
            ))
        
        return input_dict
    
    def load_annotations(self, ann_file=None):
        scans = []
        for sequence in self.sequences:
            calib = self.read_calib()
            P2 = calib["P2"]
            P3 = calib["P3"]
            T_velo_2_cam = calib["Tr"]
            proj_matrix_2 = P2 @ T_velo_2_cam
            proj_matrix_3 = P3 @ T_velo_2_cam

            voxel_base_path = os.path.join(self.ann_file, sequence)
            img_base_path = os.path.join(self.data_root, 'data_2d_raw', sequence)

            if self.load_continuous:
                id_base_path = os.path.join(self.data_root, 'data_2d_raw', sequence, 'image_00', 'data_rect', '*.png')
            else:
                id_base_path = os.path.join(self.data_root, 'data_2d_raw', sequence, 'voxels', '*.bin')
            
            for id_path in glob.glob(id_base_path):
                img_id = id_path.split("/")[-1].split(".")[0]
                img_2_path = os.path.join(img_base_path, 'image_00', 'data_rect', img_id + '.png')
                img_3_path = os.path.join(img_base_path, 'image_01', 'data_rect', img_id + '.png')
                voxel_path = os.path.join(voxel_base_path, img_id + '_1_1.npy')

                stereo_depth_path = os.path.join(self.stereo_depth_root, "sequences", sequence, img_id + '.npy')

                if not os.path.exists(voxel_path):
                    voxel_path = None
                
                scans.append(
                    {   "img_2_path": img_2_path,
                        "img_3_path": img_3_path,
                        "sequence": sequence,
                        "frame_id": img_id,
                        "P2": P2,
                        "P3": P3,
                        "T_velo_2_cam": T_velo_2_cam,
                        "proj_matrix_2": proj_matrix_2,
                        "proj_matrix_3": proj_matrix_3,
                        "voxel_path": voxel_path,
                        # "voxel_1_2_path": voxel_1_2_path,
                        "stereo_depth_path": stereo_depth_path
                    })
        
        return scans

    def get_ann_info(self, index, key='voxel_path'):
        info = self.data_infos[index][key]
        return None if info is None else np.load(info)
    
    @staticmethod
    def read_calib(calib_path=None):
        """
        Tr transforms a point from velodyne coordinates into the 
        left rectified camera coordinate system.
        In order to map a point X from the velodyne scanner to a 
        point x in the i'th image plane, you thus have to transform it like:
        x = Pi * Tr * X
        """
        P2 = np.array([
            [552.554261, 0.000000, 682.049453, 0.000000],
            [0.000000, 552.554261, 238.769549, 0.000000],
            [0.000000, 0.000000, 1.000000, 0.000000]
        ]).reshape(3, 4)

        P3 = np.array([
            [552.554261, 0.000000, 682.049453, -328.318735],
            [0.000000, 552.554261, 238.769549, 0.000000],
            [0.000000, 0.000000, 1.000000, 0.000000]
        ]).reshape(3, 4)

        cam2velo = np.array([   
            [0.04307104361, -0.08829286498, 0.995162929, 0.8043914418],
            [-0.999004371, 0.007784614041, 0.04392796942, 0.2993489574],
            [-0.01162548558, -0.9960641394, -0.08786966659, -0.1770225824],
            [0, 0, 0, 1]
        ]).reshape(4, 4)

        velo2cam = np.linalg.inv(cam2velo)
        calib_out = {}
        calib_out["P2"] = np.identity(4)  # 4x4 matrix
        calib_out["P3"] = np.identity(4)
        calib_out["P2"][:3, :4] = P2.reshape(3, 4)
        calib_out["P3"][:3, :4] = P3.reshape(3, 4)
        calib_out["Tr"] = np.identity(4)
        calib_out["Tr"][:3, :4] = velo2cam[:3, :4]
        return calib_out
    
    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)
    
    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
    
    def dynamic_baseline(self, infos):
        P3 = infos['P3']
        P2 = infos['P2']
        baseline = P3[0,3]/(-P3[0,0]) - P2[0,3]/(-P2[0,0])
        return baseline
    
def vox2pix(cam_E, cam_k, 
            vox_origin, voxel_size, 
            img_W, img_H, 
            scene_size):
    """
    compute the 2D projection of voxels centroids
    
    Parameters:
    ----------
    cam_E: 4x4
       =camera pose in case of NYUv2 dataset
       =Transformation from camera to lidar coordinate in case of SemKITTI
    cam_k: 3x3
        camera intrinsics
    vox_origin: (3,)
        world(NYU)/lidar(SemKITTI) cooridnates of the voxel at index (0, 0, 0)
    img_W: int
        image width
    img_H: int
        image height
    scene_size: (3,)
        scene size in meter: (51.2, 51.2, 6.4) for SemKITTI and (4.8, 4.8, 2.88) for NYUv2
    
    Returns
    -------
    projected_pix: (N, 2)
        Projected 2D positions of voxels
    fov_mask: (N,)
        Voxels mask indice voxels inside image's FOV 
    pix_z: (N,)
        Voxels'distance to the sensor in meter
    """
    # Compute the x, y, z bounding of the scene in meter
    vol_bnds = np.zeros((3,2))
    vol_bnds[:,0] = vox_origin
    vol_bnds[:,1] = vox_origin + np.array(scene_size)

    # Compute the voxels centroids in lidar cooridnates
    vol_dim = np.ceil((vol_bnds[:,1]- vol_bnds[:,0])/ voxel_size).copy(order='C').astype(int)
    xv, yv, zv = np.meshgrid(
            range(vol_dim[0]),
            range(vol_dim[1]),
            range(vol_dim[2]),
            indexing='ij'
          )
    vox_coords = np.concatenate([
            xv.reshape(1,-1),
            yv.reshape(1,-1),
            zv.reshape(1,-1)
          ], axis=0).astype(int).T

    # Project voxels'centroid from lidar coordinates to camera coordinates
    cam_pts = fusion.TSDFVolume.vox2world(vox_origin, vox_coords, voxel_size)
    cam_pts = fusion.rigid_transform(cam_pts, cam_E)

    # Project camera coordinates to pixel positions
    projected_pix = fusion.TSDFVolume.cam2pix(cam_pts, cam_k)
    pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]

    # Eliminate pixels outside view frustum
    pix_z = cam_pts[:, 2]
    fov_mask = np.logical_and(pix_x >= 0,
                np.logical_and(pix_x < img_W,
                np.logical_and(pix_y >= 0,
                np.logical_and(pix_y < img_H,
                pix_z > 0))))


    return projected_pix, fov_mask, pix_z

def compute_local_frustums(projected_pix, pix_z, target, img_W, img_H, dataset, n_classes, size=4):
    """
    Compute the local frustums mask and their class frequencies
    
    Parameters:
    ----------
    projected_pix: (N, 2)
        2D projected pix of all voxels
    pix_z: (N,)
        Distance of the camera sensor to voxels
    target: (H, W, D)
        Voxelized sematic labels
    img_W: int
        Image width
    img_H: int
        Image height
    dataset: str
        ="NYU" or "kitti" (for both SemKITTI and KITTI-360)
    n_classes: int
        Number of classes (12 for NYU and 20 for SemKITTI)
    size: int
        determine the number of local frustums i.e. size * size
    
    Returns
    -------
    frustums_masks: (n_frustums, N)
        List of frustums_masks, each indicates the belonging voxels  
    frustums_class_dists: (n_frustums, n_classes)
        Contains the class frequencies in each frustum
    """
    H, W, D = target.shape
    ranges = [(i * 1.0/size, (i * 1.0 + 1)/size) for i in range(size)]
    local_frustum_masks = []
    local_frustum_class_dists = []
    pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]
    for y in ranges:
        for x in ranges:
            start_x = x[0] * img_W
            end_x = x[1] * img_W
            start_y = y[0] * img_H
            end_y = y[1] * img_H
            
            def compute_local_frustum(pix_x, pix_y, min_x, max_x, min_y, max_y, pix_z):
                valid_pix = np.logical_and(pix_x >= min_x,
                            np.logical_and(pix_x < max_x,
                            np.logical_and(pix_y >= min_y,
                            np.logical_and(pix_y < max_y,
                            pix_z > 0))))
                return valid_pix
            
            local_frustum = compute_local_frustum(pix_x, pix_y, start_x, end_x, start_y, end_y, pix_z)
            if dataset == "NYU":
                mask = (target != 255) & np.moveaxis(local_frustum.reshape(60, 60, 36), [0, 1, 2], [0, 2, 1])
            elif dataset == "kitti":
                mask = (target != 255) & local_frustum.reshape(H, W, D)

            local_frustum_masks.append(mask)
            classes, cnts = np.unique(target[mask], return_counts=True)
            class_counts = np.zeros(n_classes)
            class_counts[classes.astype(int)] = cnts
            local_frustum_class_dists.append(class_counts)
    frustums_masks, frustums_class_dists = np.array(local_frustum_masks), np.array(local_frustum_class_dists)
    return frustums_masks, frustums_class_dists

def compute_CP_mega_matrix(target, is_binary=False):
    """
    Parameters
    ---------
    target: (H, W, D)
        contains voxels semantic labels

    is_binary: bool
        if True, return binary voxels relations else return 4-way relations
    """
    label = target.reshape(-1)
    label_row = label
    N = label.shape[0]
    super_voxel_size = [i//2 for i in target.shape]
    if is_binary:
        matrix = np.zeros((2, N, super_voxel_size[0] * super_voxel_size[1] * super_voxel_size[2]), dtype=np.uint8)
    else:
        matrix = np.zeros((4, N, super_voxel_size[0] * super_voxel_size[1] * super_voxel_size[2]), dtype=np.uint8)

    for xx in range(super_voxel_size[0]):
        for yy in range(super_voxel_size[1]):
            for zz in range(super_voxel_size[2]):
                col_idx = xx * (super_voxel_size[1] * super_voxel_size[2]) + yy * super_voxel_size[2] + zz
                label_col_megas = np.array([
                    target[xx * 2,     yy * 2,     zz * 2],
                    target[xx * 2 + 1, yy * 2,     zz * 2],
                    target[xx * 2,     yy * 2 + 1, zz * 2],
                    target[xx * 2,     yy * 2,     zz * 2 + 1],
                    target[xx * 2 + 1, yy * 2 + 1, zz * 2],
                    target[xx * 2 + 1, yy * 2,     zz * 2 + 1],
                    target[xx * 2,     yy * 2 + 1, zz * 2 + 1],
                    target[xx * 2 + 1, yy * 2 + 1, zz * 2 + 1],
                ])
                label_col_megas = label_col_megas[label_col_megas != 255]
                for label_col_mega in label_col_megas:
                    label_col = np.ones(N)  * label_col_mega
                    if not is_binary:
                        matrix[0, (label_row != 255) & (label_col == label_row) & (label_col != 0), col_idx] = 1.0 # non non same
                        matrix[1, (label_row != 255) & (label_col != label_row) & (label_col != 0) & (label_row != 0), col_idx] = 1.0 # non non diff
                        matrix[2, (label_row != 255) & (label_row == label_col) & (label_col == 0), col_idx] = 1.0 # empty empty
                        matrix[3, (label_row != 255) & (label_row != label_col) & ((label_row == 0) | (label_col == 0)), col_idx] = 1.0 # nonempty empty
                    else:
                        matrix[0, (label_row != 255) & (label_col != label_row), col_idx] = 1.0 # diff
                        matrix[1, (label_row != 255) & (label_col == label_row), col_idx] = 1.0 # same
    return matrix