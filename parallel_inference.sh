#! /bin/bash

YAML_NAME=gen_sg_v1_resnetweight08.yaml
EXP_NAME=gen_v1_resnet08
NAME=last08-
N_SPLIT=20

for ((split_idx=0; split_idx<N_SPLIT; split_idx++))
do
    echo $split_idx
    brain task create -p 76 --cmd "python aux_main.py --config_file configs/${YAML_NAME} --exp_name ${EXP_NAME} --inference --num_workers 32 --split_idx ${split_idx} --n_split ${N_SPLIT}" -n ${NAME}${split_idx} -t "braincloud-v2-kakaobrain-woong.ssang_kakaobrain.com-797b94b5-4365-4e64-87ab-12dfdf15450f" -f v1.xlarge -w /data/project/rw/woong.ssang/CBIR/
done

