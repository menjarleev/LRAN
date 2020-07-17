#!/usr/bin/env bash
python ../main.py --save_result --dataset_root /home/menjarleev/PycharmProjects/dataset/DIV2K/train --dataroot_test /home/menjarleev/PycharmProjects/dataset/Set14 --name INSR --loss_term l1 --dataset DIV2K_SR --gpu_id 0 --netG INSR --amp O0
