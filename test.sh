CUDA_VISIBLE_DEVICES=0 \
python main.py \
--eval \
--ckpt_path logs/SimpleFormer-Resnet101-imgnet-Swin-KITTI360-unified-LSS/tensorboard/version_0/checkpoints/best.ckpt \
--config_path logs/SimpleFormer-Resnet101-imgnet-Swin-KITTI360-unified-LSS/config.py \
--log_folder version3 \
--seed 7240 \
--log_every_n_steps 100