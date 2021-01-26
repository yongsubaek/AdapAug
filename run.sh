#!/bin/bash

gpu=
cv_id=0
n_mismatch=0
n_controller=1
aw=2
dw=1

. ./parse_options.sh

if [ -z ${gpu} ]; then
    echo "Error: set GPU number." 1>&2
    echo "Usage: ./run.sh --gpu 0" 1>&2
    exit 1
fi

echo gpu: ${gpu}
echo cv_id: ${cv_id}
echo n_mismatch: ${n_mismatch}
echo n_controller: ${n_controller}
echo exp_name: c10svhn_v3_affdivorder_${n_controller}_${n_mismatch}_${cv_id}_${aw}_${dw}
echo aw: ${aw}
echo dw: ${dw}

CUDA_VISIBLE_DEVICES=${gpu} python AdapAug/elem_search.py \
	--config confs/wresnet40x2_cifar_svhn.yaml \
	--dataroot ~/data \
	--cv-num 5 \
	--childaug default \
	--aug default \
	--version 3 \
	--aw ${aw} \
	--dw ${dw} \
    --ew 1e-5 \
	--r_type 4 \
	--num-policy 5 \
	--cv_id ${cv_id} \
	--M 1 \
	--mode reinforce \
	--lstm_n 1 \
	--exp_name c10svhn_v3_affdivorder_${n_mismatch}_${cv_id} \
	--n_mismatch ${n_mismatch} \
    --n_controller ${n_controller} \
	--no_img
