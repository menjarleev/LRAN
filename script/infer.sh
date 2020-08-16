#!/usr/bin/env bash
# PRAN
user_name="menjarleev";
gpu_id="0";
num_blocks="125";
group_size="5";
name="exp2"
dataset="Set14"
pretrain="~/PycharmProjects/LRAN/script/ckpt/exp2/state_Set14.pth"

python ../main.py --save_result --dataset Set14 --name exp2 --loss_term l1  --gpu_id 0 --netG PRAN --group_size 5 --num_blocks 125 --pretrain /home/menjarleev/PycharmProjects/LRAN/script/ckpt/exp2/state_Set14.pth --infer --infer_root /home/menjarleev/PycharmProjects/dataset/SR/LR/LRBI/Set14

