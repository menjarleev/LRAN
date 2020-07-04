#!/usr/bin/env bash
#python main.py --save_result --dataset_root /home/menjarleev/PycharmProjects/dataset/DIV2K/train --name GAN --loss_term GAN
python main.py --save_result --dataset_root /home/lhuo9710/dataset/AIM/train --name AIM --loss_term GAN --gpu_id 0 --augs cutblur
