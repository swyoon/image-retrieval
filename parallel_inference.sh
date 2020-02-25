#! /bin/bash
YAML_NAME=han_sbert_f30k.yaml
EXP_NAME=han_f30k_2
NAME=han-f30k
N_SPLIT=10
EPOCH=25

for ((split_idx=0; split_idx<N_SPLIT; split_idx++))
do
    echo $split_idx
    brain task create -p 76 --cmd "python main_woong.py --config_file configs/${YAML_NAME} --exp_name ${EXP_NAME} --inference --split_idx ${split_idx} --n_split ${N_SPLIT} " -n ${NAME}${split_idx} -t "braincloud-v2-kakaobrain-woong.ssang_kakaobrain.com-96f72c60-7ad3-45a4-a087-2cf5621c876a" -f p1.large -w /data/project/rw/woong.ssang/CBIR/ -z public:normal
done

