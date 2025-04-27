import mmcv
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from mmdet.datasets.builder import PIPELINES
import math

@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """
    def __init__(self, 
            data_config,
            is_train=False,
            img_norm_cfg=None,
            load_stereo_depth=False,
            color_jitter=(0.4, 0.4, 0.4),
            znear=1.0,
            zfar=50,
        ):
        super().__init__()

        self.is_train = is_train
        self.data_config = data_config
        self.img_norm_cfg = img_norm_cfg

        self.load_stereo_depth = load_stereo_depth
        self.color_jitter = (
            transforms.ColorJitter(*color_jitter) if color_jitter else None
        )

        self.normalize_img = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.ToTensor = transforms.ToTensor()
        self.znear = znear
        self.zfar = zfar
    

    def get_rot(self,h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def sample_augmentation(self, H , W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        
        if self.is_train:
            resize = float(fW)/float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])

        else:
            resize = float(fW) / float(W)
            resize += self.data_config.get('resize_test', 0.0)
            if scale is not None:
                resize = scale

            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0

        return resize, resize_dims, crop, flip, rotate

    def img_transform(self, img, post_rot, post_tran,
                      resize, resize_dims, crop,
                      flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        
        return img
    
    def get_inputs(self, results, flip=None, scale=None):
        img_filenames = results['img_filename']

        focal_length = results['focal_length']

        data_lists = []
        raw_img_list = []
        lidar2img_rts = []
        fovx_list = []
        fovy_list = []
        viewmatrix_list = []
        projmatrix_list = []
        cam_pos_list = []
        for i in range(len(img_filenames)):
            img_filename = img_filenames[i]
            img = Image.open(img_filename).convert('RGB')

            # perform image-view augmentation
            post_rot = torch.eye(2)
            post_trans = torch.zeros(2)

            if i == 0:
                img_augs = self.sample_augmentation(H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs
            img, post_rot2, post_tran2 = self.img_transform(
                img, post_rot, post_trans, resize=resize, 
                resize_dims=resize_dims, crop=crop, flip=flip, rotate=rotate
            )

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            # intrins
            intrin = torch.Tensor(results['cam_intrinsic'][i])

            # extrins
            lidar2cam = torch.Tensor(results['lidar2cam'][i])
            cam2lidar = lidar2cam.inverse()
            rot = cam2lidar[:3, :3]
            tran = cam2lidar[:3, 3]

            # output
            canvas = np.array(img)
            
            fovx = self._focal2fov(intrin[0, 0], img.width)
            fovy = self._focal2fov(intrin[1, 1], img.height)
            
            viewmatrix = self._getWorld2View2(np.transpose(lidar2cam[:3, :3]), lidar2cam[:3, 3], translate=np.array([0.0, 0.0, 0.0]), scale=1.0)
            viewmatrix = torch.tensor(viewmatrix, dtype=torch.float32).transpose(0, 1)
            
            projmatrix_ = self._getProjectionMatrixShift(self.znear, 
                                                             self.zfar, 
                                                             intrin[0, 0], 
                                                             intrin[1, 1], 
                                                             intrin[0, 2], 
                                                             intrin[1, 2], 
                                                             img.width, img.height, fovx, fovy)
            projmatrix_ = projmatrix_.transpose(0,1)
            full_proj_transform = (viewmatrix.unsqueeze(0).bmm(projmatrix_.unsqueeze(0))).squeeze(0)
            
            cam_pos = viewmatrix.inverse()[3, :3]
            
            if self.color_jitter and self.is_train:
                img = self.color_jitter(img)
            
            img = self.normalize_img(img)

            result = [img, rot, tran, intrin, post_rot, post_tran, cam2lidar]
            result = [x[None] for x in result]

            data_lists.append(result)
            raw_img_list.append(canvas)
            
            lidar2img_rts.append(intrin @ lidar2cam)
            fovx_list.append(fovx)
            fovy_list.append(fovy)
            viewmatrix_list.append(viewmatrix)
            projmatrix_list.append(full_proj_transform)
            cam_pos_list.append(cam_pos)
        
        if self.load_stereo_depth:
            stereo_depth_path = results['stereo_depth_path']
            stereo_depth = np.load(stereo_depth_path)
            stereo_depth = Image.fromarray(stereo_depth)
            resize, resize_dims, crop, flip, rotate = img_augs
            stereo_depth = self.img_transform_core(stereo_depth, resize_dims=resize_dims,
                    crop=crop, flip=flip, rotate=rotate)
            results['stereo_depth'] = self.ToTensor(stereo_depth)
        num = len(data_lists[0])
        result_list = []
        for i in range(num):
            result_list.append(torch.cat([x[i] for x in data_lists], dim=0))
        
        results['focal_length'] = torch.tensor(focal_length, dtype=torch.float32)
        results['raw_img'] = raw_img_list
        results['lidar2img'] = lidar2img_rts
        results['fovx'] = fovx_list
        results['fovy'] = fovy_list
        results['viewmatrix'] = viewmatrix_list
        results['projmatrix'] = projmatrix_list
        results['cam_pos'] = cam_pos_list

        return result_list

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)

        return results
    
    def _focal2fov(self, focal, pixel):
        # return 2 * math.atan(pixel / (2 * focal)) * (180 / np.pi) 
        return 2 * math.atan(pixel / (2 * focal))
    
    def _getWorld2View2(self, R, t, translate=np.array([.0, .0, .0]), scale=1.0):
        # 用于从 w2c 中提取的 R 和 t，得到 平移和缩放后的 c2w
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R.T
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0

        C2W = np.linalg.inv(Rt)
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale
        C2W[:3, 3] = cam_center
        Rt = np.linalg.inv(C2W)
        return np.float32(Rt)
    
    def _getProjectionMatrix(self, znear, zfar, fovX, fovY):
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = np.zeros((4, 4))

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P
    
    def _getProjectionMatrixShift(self, znear, zfar, focal_x, focal_y, cx, cy, width, height, fovX, fovY):
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))

        # the origin at center of image plane
        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        # shift the frame window due to the non-zero principle point offsets
        offset_x = cx - (width/2)
        offset_x = (offset_x/focal_x)*znear
        offset_y = cy - (height/2)
        offset_y = (offset_y/focal_y)*znear

        top = top + offset_y
        left = left + offset_x
        right = right + offset_x
        bottom = bottom + offset_y

        P = torch.zeros(4, 4)

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P

@PIPELINES.register_module()
class LoadCameraParam(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
        self,
        cam_nums=1,
        znear=1.0,
        zfar=50,
    ):
        self.cam_nums = cam_nums
        self.znear = znear
        self.zfar = zfar
    
    def _focal2fov(self, focal, pixel):
        # return 2 * math.atan(pixel / (2 * focal)) * (180 / np.pi) 
        return 2 * math.atan(pixel / (2 * focal))

    def _getProjectionMatrix(self, znear, zfar, fovX, fovY):
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = np.zeros((4, 4))

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P


    def _getProjectionMatrixShift(self, znear, zfar, focal_x, focal_y, cx, cy, width, height, fovX, fovY):
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))

        # the origin at center of image plane
        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        # shift the frame window due to the non-zero principle point offsets
        offset_x = cx - (width/2)
        offset_x = (offset_x/focal_x)*znear
        offset_y = cy - (height/2)
        offset_y = (offset_y/focal_y)*znear

        top = top + offset_y
        left = left + offset_x
        right = right + offset_x
        bottom = bottom + offset_y

        P = torch.zeros(4, 4)

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P

    def _getWorld2View2(self, R, t, translate=np.array([.0, .0, .0]), scale=1.0):
        # 用于从 w2c 中提取的 R 和 t，得到 平移和缩放后的 c2w
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R.transpose()
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0

        C2W = np.linalg.inv(Rt)
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale
        C2W[:3, 3] = cam_center
        Rt = np.linalg.inv(C2W)
        return np.float32(Rt)


    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """

        cam_params = []
        for cam_idx in range(self.cam_nums):
            height = results['img_shape'][0]
            width = results['img_shape'][1]

            #TODO
            if isinstance(results['cam_intrinsic'][cam_idx], list) or results['cam_intrinsic'][cam_idx].ndim >=3:
                fovx = self._focal2fov(results['cam_intrinsic'][cam_idx][0][0, 0], width)
                fovy = self._focal2fov(results['cam_intrinsic'][cam_idx][0][1, 1], height)

                w2c = results['lidar2cam'][cam_idx][0]
                viewmatrix = self._getWorld2View2(np.transpose(w2c[:3, :3]), w2c[:3, 3], translate=np.array([0.0, 0.0, 0.0]), scale=1.0)
                viewmatrix = torch.tensor(viewmatrix, dtype=torch.float32).transpose(0, 1)
                projmatrix_ = self._getProjectionMatrixShift(self.znear, 
                                                             self.zfar, 
                                                             results['cam_intrinsic'][cam_idx][0][0, 0], 
                                                             results['cam_intrinsic'][cam_idx][0][1, 1], 
                                                             results['cam_intrinsic'][cam_idx][0][0, 2], 
                                                             results['cam_intrinsic'][cam_idx][0][1, 2], 
                                                             width, height, fovx, fovy)
                projmatrix_ = projmatrix_.transpose(0,1)

            else:
                fovx = self._focal2fov(results['cam_intrinsic'][cam_idx][0, 0], width)
                fovy = self._focal2fov(results['cam_intrinsic'][cam_idx][1, 1], height)

                w2c = results['lidar2cam'][cam_idx]
                viewmatrix = self._getWorld2View2(np.transpose(w2c[:3, :3]), w2c[:3, 3], translate=np.array([0.0, 0.0, 0.0]), scale=1.0)
                viewmatrix = torch.tensor(viewmatrix, dtype=torch.float32).transpose(0, 1)
                
                projmatrix_ = self._getProjectionMatrixShift(self.znear, 
                                                             self.zfar, 
                                                             results['cam_intrinsic'][cam_idx][0, 0], 
                                                             results['cam_intrinsic'][cam_idx][1, 1], 
                                                             results['cam_intrinsic'][cam_idx][0, 2], 
                                                             results['cam_intrinsic'][cam_idx][1, 2], 
                                                             width, height, fovx, fovy)
                projmatrix_ = torch.tensor(projmatrix_, dtype=torch.float32).transpose(0,1)

            # # w2i
            full_proj_transform = (viewmatrix.unsqueeze(0).bmm(projmatrix_.unsqueeze(0))).squeeze(0)

            cam_pos = viewmatrix.inverse()[3, :3]
            cam_param = {'height': height, 
                         'width':width, 
                         'fovx':fovx, 
                         'fovy':fovy, 
                         'viewmatrix':viewmatrix, 
                         'projmatrix':full_proj_transform, 
                         'cam_pos':cam_pos}
            cam_params.append(cam_param)
        
        results['cam_params'] = cam_params

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        repr_str += f"color_type='{self.color_type}')"
        return repr_str
