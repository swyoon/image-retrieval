#! /bin/bash

YAML_NAME=v2_meanbsert.yaml
EXP_NAME=v2_meansbert
NAME=v2-meansbert-
N_SPLIT=20
EPOCH=13

for ((split_idx=0; split_idx<N_SPLIT; split_idx++))
do
    echo $split_idx
    brain task create -p 76 --cmd "python aux_main.py --config_file configs/${YAML_NAME} --exp_name ${EXP_NAME} --inference --num_workers 24 --split_idx ${split_idx} --n_split ${N_SPLIT} --epoch ${EPOCH}" -n ${NAME}${split_idx} -t "braincloud-v2-kakaobrain-woong.ssang_kakaobrain.com-797b94b5-4365-4e64-87ab-12dfdf15450f" -f p2.xlarge -w /data/project/rw/woong.ssang/CBIR/
done

