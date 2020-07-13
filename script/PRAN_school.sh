#!/usr/bin/env bash
user_name="lhuo9710";
gpu_id="2";
python ../main.py --save_result --dataset_root /home/$user_name/PycharmProjects/dataset/DIV2K/train --dataroot_test /home/$user_name/PycharmProjects/dataset/Set14 --name PRAN --loss_term l1 --dataset DIV2K_SR --gpu_id $gpu_id --netG PRAN
