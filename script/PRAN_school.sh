#!/usr/bin/env bash
# PRAN
user_name="lhuo9710";
gpu_id="3";
num_blocks="125";
group_size="5";
name="exp2"

#user_name="lhuo9710";
#gpu_id="2";
#num_blocks="243";
#group_size="3";

python ../main.py --save_result --dataset_root /home/$user_name/PycharmProjects/dataset/DIV2K/train --dataroot_test /home/$user_name/PycharmProjects/dataset/Set14 --name $name --loss_term l1 --dataset DIV2K_SR --gpu_id $gpu_id --netG PRAN --amp O0
