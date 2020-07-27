#!/usr/bin/env bash
# exp with body with receptive field of 51
python ../main.py --save_result --dataset_root /home/lhuo9710/PycharmProjects/dataset/DIV2K/train --dataroot_test /home/lhuo9710/PycharmProjects/dataset/Set14 --name srrcan --loss_term l1 --dataset DIV2K_SR --gpu_id 0 --netG SRRCAN
