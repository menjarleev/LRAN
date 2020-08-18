#!/usr/bin/env bash
python ../main.py --save_result --root /home/lhuo9710/PycharmProjects/dataset/DIV2K/train --dataroot_test /home/lhuo9710/PycharmProjects/dataset/Set14 --name rcan --loss_term l1 --dataset DIV2K_SR --gpu_id 3 --netG RCAN
