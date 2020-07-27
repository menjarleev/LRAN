#!/usr/bin/env bash
python ../main.py --save_result --dataset_root /home/lhuo9710/PycharmProjects/dataset/DIV2K/train --dataroot_test /home/lhuo9710/PycharmProjects/dataset/Set14 --name LSR --loss_term l1 --dataset DIV2K_SR --gpu_id 2 --netG LSR --amp O0
