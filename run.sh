#!/bin/bash
python AdapAug/elem_search.py \
	--config confs/wresnet40x2_cifar_svhn.yaml \
	--dataroot ../data \
	--cv-num 5 \
	--childaug default \
	--aug default \
	--version 3 \
	--aw 2 \
	--dw 1 \
	--r_type 4 \
	--num-policy 5 \
	--cv_id 0 \
	--M 1 \
	--mode reinforce \
	--lstm_n 1 \
	--exp_name c10svhn_v3_affdivorder \
	--n_mismatch 0 \
	--no_img
