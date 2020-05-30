#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python main_woong.py --config_file configs/vgsp_gt_ssgpoolV2.yaml --exp_name ssghard_inv_l1r1_jump_nospec --tb_path tb/ \
     --ckpt_path ckpt/
