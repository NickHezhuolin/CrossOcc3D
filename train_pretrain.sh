# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# nohup python main.py \
# --config_path configs/CGFormer-Efficient-Swin-KITTI360-unified.py \
# --log_folder CGFormer-Efficient-Swin-KITTI360-unified \
# --seed 7240 \
# --pretrain \
# --log_every_n_steps 100 \
# > CGFormer-Efficient-Swin-KITTI360-unified.log 2>&1 &

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main.py \
--config_path configs/SimperGaussianPretrain-Resnet101.py \
--log_folder SimperGaussianPretrain-Resnet101 \
--seed 7240 \
--pretrain \
--log_every_n_steps 100
# > SimpleFormer-Resnet101-Swin-KITTI360-unified-LSS 2>&1