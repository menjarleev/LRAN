#!/usr/bin/env bash
python ../main.py --save_result --root /home/menjarleev/PycharmProjects/dataset/DIV2K/train --dataroot_test /home/menjarleev/PycharmProjects/dataset/Set14 --name LSR --loss_term l1 --dataset DIV2K_SR --gpu_id 0 --netG LSR --amp O1
