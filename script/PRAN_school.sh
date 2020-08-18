#!/usr/bin/env bash
# PRAN
user_name="lhuo9710";
gpu_id="3";
num_blocks="125";
group_size="5";
name="PRANB125G5C64"
num_channels="64";

python ../main.py --save_result --root /home/$user_name/PycharmProjects/dataset/SR --name $name --loss_term l1 --gpu_id $gpu_id --netG PRAN --group_size $group_size --num_blocks $num_blocks --num_channels $num_channels
