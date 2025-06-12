CUDA_VISIBLE_DEVICES=0 \
python main.py \
--eval \
--ckpt_path logs/SimpleFormer-Resnet101-Waymo-unified-LSS-0608/tensorboard/version_0/checkpoints/epoch-epoch=28.ckpt \
--config_path logs/SimpleFormer-Resnet101-Waymo-unified-LSS-0608/config.py \
--log_folder version0 \
--seed 7240 \
--log_every_n_steps 100