#! /bin/bash

CUDA_VISIBLE_DEVICES=3 python main_img.py --config_file configs/resnet_vg_reg.yaml --exp_name resnet_vg_reg_test --tb_path tb/ \
    --result_path results/ --ckpt_path ckpt/
