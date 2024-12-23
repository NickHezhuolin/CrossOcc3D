CUDA_VISIBLE_DEVICES=0 \
nohup python main.py \
--eval \
--ckpt_path ./logs/CGFormer-Efficient-Swin-KITTI360-unified/tensorboard/version_4/checkpoints/best.ckpt \
--config_path configs/CGFormer-Efficient-Swin-KITTI360-unified.py \
--log_folder version4 \
--seed 7240 \
--log_every_n_steps 100 \
> CCGFormer-Efficient-Swin-KITTI360-unified-Eval.log 2>&1 &