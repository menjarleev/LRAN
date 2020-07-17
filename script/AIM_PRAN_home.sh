#!/usr/bin/env bash
user="menjarleev";
name="AIM_PRAN_home";
python ../main.py --save_result --dataset_root /home/$user/PycharmProjects/dataset/AIM/train --name $name --loss_terms l1 --gpu_id 0 --augs cutblur --no_test_during_train --eval_metric score --netG PRAN --amp O0
