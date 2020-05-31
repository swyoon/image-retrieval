#! /bin/bash

# CUDA_VISIBLE_DEVICES=2 python main_woong.py --config_file configs/han_meansbert_gen_repro.yaml \
#     --exp_name vg_han_tmb --ckpt ckpt/ --tb_path tb/ --max_epoch 25

YAML_NAME=vgsp_sagpool08.yaml
EXP_NAME=vgsp_sagpool08
NAME=vgsp-sagpool08
CMD="python main_woong.py --config_file configs/${YAML_NAME} --exp_name ${EXP_NAME} --max_epoch 25 --num_workers 14"
# CUDA_VISIBLE_DEVICES=2 $CMD
brain task create -p 76 --cmd "${CMD}" \
    -n ${NAME}${split_idx} -t "braincloud-v2-kakaobrain-woong.ssang_kakaobrain.com-217f7578-64be-4ce5-a5e8-e262197c13cb"\
    -f v1.xlarge -w /data/project/rw/woong.ssang/CBIR/ -z private:normal --http-proxy-on


