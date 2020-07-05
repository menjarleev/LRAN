#!/usr/bin/env bash
python main.py --save_result --dataset_root /home/lhuo9710/dataset/AIM/train --name AIM --loss_terms l1,gan,vgg --gpu_id 0 --augs cutblur --no_test_during_train --continue_train
