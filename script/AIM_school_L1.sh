#!/usr/bin/env bash
python ../main.py --save_result --dataset_root /home/lhuo9710/PycharmProjects/dataset/AIM/train --name L1_AIM --loss_terms l1 --gpu_id 0 --augs cutblur --no_test_during_train --normalize --eval_metric score

