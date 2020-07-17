#!/usr/bin/env bash
python ../main.py --save_result --name AIM_infer --gpu_id 3 --netG PRAN --infer --pretrain /home/lhuo9710/PycharmProjects/LRAN/script/ckpt/PRAN_AIM/state_latest.pth --augs cutblur

