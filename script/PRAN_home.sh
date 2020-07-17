#!/usr/bin/env bash
user="menjarleev";
name="div2k_PRAN_vgg";
python ../main.py --save_result --dataset_root /home/menjarleev/PycharmProjects/dataset/DIV2K/train --dataroot_test /home/menjarleev/PycharmProjects/dataset/Set14 --name PRAN --loss_term l1 --dataset DIV2K_SR --gpu_id 0 --netG PRAN --amp O0
