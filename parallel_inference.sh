#! /bin/bash
YAML_NAME=vg_ssgpool20.yaml
EXP_NAME=vg_ssgpool20
NAME=vg-ssgpool20-
N_SPLIT=14
EPOCH=24

for ((split_idx=0; split_idx<N_SPLIT; split_idx++))
do
    echo $split_idx
    brain task create -p 76 --cmd "python main_woong.py --config_file configs/${YAML_NAME} --exp_name ${EXP_NAME} --inference --split_idx ${split_idx} --n_split ${N_SPLIT} --epoch ${EPOCH} --json --num_workers 6"\
        -n ${NAME}${split_idx} -t "braincloud-v2-kakaobrain-woong.ssang_kakaobrain.com-cb164000-191e-4bf5-bc1c-be68829568d3" -f v1.xlarge -w /data/project/rw/woong.ssang/CBIR/ -z private:normal\
        --http-proxy-on
done

