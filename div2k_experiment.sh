#!/usr/bin/env bash
python main.py --save_result --dataset_root /home/lhuo9710/PycharmProjects/dataset/DIV2K/train --dataroot_test /home/lhuo9710/PycharmProjects/dataset/Set14 --name exp --loss_term l1,vgg,gan --dataset DIV2K_SR --gpu_id 3
