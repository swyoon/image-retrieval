#! /bin/bash

YAML_NAME=vgsp_sagpool005_ep30.yaml
EXP_NAME=vgsp_sagpool005_ep30
NAME=vgsp-sagpool-
N_SPLIT=5
EPOCH=29

for ((split_idx=0; split_idx<N_SPLIT; split_idx++))
do
    echo $split_idx
    # brain task create -p 76 --cmd "python main_woong.py --config_file configs/${YAML_NAME} --exp_name ${EXP_NAME} --inference --split_idx ${split_idx} --n_split ${N_SPLIT} --epoch ${EPOCH} --json --num_workers 6"\
    brain task create -p 76 --cmd "python main_woong.py --config_file configs/${YAML_NAME} --exp_name ${EXP_NAME} --inference --split_idx ${split_idx} --n_split ${N_SPLIT} --epoch ${EPOCH} --num_workers 6 --infer_batch"\
       -n ${NAME}${split_idx} -t "braincloud-v2-kakaobrain-woong.ssang_kakaobrain.com-cb164000-191e-4bf5-bc1c-be68829568d3" -f p1.xlarge -w /data/project/rw/woong.ssang/CBIR/ -z private:normal\
       --http-proxy-on
done
