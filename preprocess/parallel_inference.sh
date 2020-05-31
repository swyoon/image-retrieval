#! /bin/bash
NAME=vg-coco-tr
N_SPLIT=30

for ((split_idx=0; split_idx<N_SPLIT; split_idx++))
do
    echo $split_idx
    brain task create -p 76 --cmd "python prepare_viewer_input.py vg_coco train ${split_idx} ${N_SPLIT}" -n ${NAME}${split_idx} -t "braincloud-v2-kakaobrain-woong.ssang_kakaobrain.com-96f72c60-7ad3-45a4-a087-2cf5621c876a" -f cg.small -w /data/project/rw/woong.ssang/CBIR/preprocess -z private:normal
done

