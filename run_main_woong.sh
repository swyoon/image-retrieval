#! /bin/bash

# CUDA_VISIBLE_DEVICES=2 python main_woong.py --config_file configs/han_meansbert_gen_repro.yaml \
#     --exp_name vg_han_tmb --ckpt ckpt/ --tb_path tb/ --max_epoch 25

YAML_NAME=coco_han_tmb.yaml
EXP_NAME=coco_han_tmb
NAME=coco-han-tmb
brain task create -p 76 --cmd "python main_woong.py --config_file configs/${YAML_NAME} --exp_name ${EXP_NAME} --max_epoch 40" -n ${NAME}${split_idx} -t "braincloud-v2-kakaobrain-woong.ssang_kakaobrain.com-96f72c60-7ad3-45a4-a087-2cf5621c876a" -f v2.xlarge -w /data/project/rw/woong.ssang/CBIR/ -z private:normal


