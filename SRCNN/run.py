#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 20:17:13 2023

@author: sofiahernandezgelado
"""

eval_dir = "/home/daniel/Documents/SRCNN_pytorch2/eval"
train_dir = "/home/daniel/Documents/SRCNN_pytorch2/train"
ouput_dir = "/home/daniel/Documents/SRCNN_pytorch2/Exp2/outputs"


train_file = train_dir + "/train_file.h5"
eval_file = eval_dir + "/eval_file.h5"

!python /home/sofiahernandezgelado/Documents/SRCNN_pytorch2/prepare.py\
    --images-dir "/home/sofiahernandezgelado/Documents/SRCNN_pytorch2/LargeTrial/train"\
    --output-path "/home/sofiahernandezgelado/Documents/SRCNN_pytorch2/LargeTrial/train_file.h5"\
    --scale 8


!python /home/sofiahernandezgelado/Documents/SRCNN_pytorch2/prepare.py\
--images-dir "/home/sofiahernandezgelado/Documents/SRCNN_pytorch2/LargeTrial/eval"\
--output-path "/home/sofiahernandezgelado/Documents/SRCNN_pytorch2/LargeTrial/eval_file.h5"\
--scale 8 --eval


#"/home/daniel/Documents/SRCNN_pytorch2/train/train_file.h5"
#"/home/daniel/Documents/SRCNN_pytorch2/eval/eval_file.h5"
#/home/daniel/Documents/SRCNN_pytorch2/Experiments/Exp4/outputs"

!python /home/daniel/Documents/SRCNN_pytorch2/train.py\
    --train-file "/home/daniel/Documents/SRCNN_pytorch2/train/train_file.h5" \
                --eval-file "/home/daniel/Documents/SRCNN_pytorch2/eval/eval_file.h5" \
                --outputs-dir "/home/daniel/Documents/SRCNN_pytorch2/Experiments/Exp6/outputs" \
                --scale 2 \
                --lr 1e-6 \
                --batch-size 32\
                --num-epochs 100\
                --num-workers 2 \
                --seed 123
                
             
               
!python /home/daniel/Documents/SRCNN_pytorch2/test.py \
--weights-file "/home/daniel/Documents/SRCNN_pytorch2/Experiments/Exp7/outputs/x4/epoch_723.pth" \
--image-dir "/home/daniel/Documents/SRCNN_pytorch2/Experiments/Exp7/test" \
--scale 4
               


!python /home/daniel/Documents/SRCNN_pytorch2/test.py \
--weights-file "/home/daniel/Documents/Test1/outputs/x4/best.pth" \
--image-dir "/home/daniel/Documents/Test1/test" \
--scale 4
            

                
import torch 
torch.cuda.is_available()


!python /home/daniel/Documents/pytorch-vdsr/main_vdsr.py --cuda
