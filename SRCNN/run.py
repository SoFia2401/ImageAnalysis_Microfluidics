#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 20:17:13 2023

@author: sofiahernandezgelado
"""

# Change directory to repository
import os
os.chdir("/home/SuperResolution_Microfluidics")

# Prepare h5py file for train
!python /SRCNN/prepare.py\
    --images-dir "data/train"\
    --output-path "data/train/train_file_SRCNN.h5"\
    --scale 2

# Prepare h5py file for eval
!python /SRCNN/prepare.py\
--images-dir "data/eval"\
--output-path "data/eval/eval_file_SRCNN.h5"\
--scale 2 --eval

# Start training
!python  /SRCNN/train.py\
    --train-file "data/train/train_file_SRCNN.h5" \
                --eval-file "data/eval/eval_file_SRCNN.h5" \
                --outputs-dir "SRCNN/outputs" \
                --scale 2 \
                --lr 1e-5 \
                --batch-size 32\
                --num-epochs 100\
                --num-workers 2 \
                --seed 123
                           
# Test performance                
!python /SRCNN/test.py\
--image-dir "data/test"\
--weights-file "SRCNN/LargeTrial/Checkpoints and results/x2/best.pth"\
--scale 2