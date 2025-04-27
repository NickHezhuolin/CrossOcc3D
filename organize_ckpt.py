import torch
from argparse import ArgumentParser




def parse_config():
    parser = ArgumentParser()
    parser.add_argument('--source_path', default='/home/hez4sgh/1_work_dir/CGFormer/logs/SimperGaussianPretrain-Resnet101/tensorboard/version_1/checkpoints/epoch-epoch=19.ckpt')
    parser.add_argument('--dst_path', default='ckpts/gs-resnet101-kitti360-cgformer-learning-setting.pth')
    
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_config()

    checkpoints = torch.load(args.source_path, map_location='cpu')['state_dict']
    new_checkpoints = {}
    for key in checkpoints:
        new_checkpoints[key.replace('model.', '')] = checkpoints[key]
    
    torch.save({'state_dict': new_checkpoints}, args.dst_path)