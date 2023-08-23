#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 19:08:51 2023

@author: sofiahernandezgelado
"""
import os
os.chdir("/home/sofiahernandezgelado/Documents/super-image/super_image/src")

#from datasets import load_dataset
from super_image.data import EvalDatasetH5, TrainAugmentDatasetH5, augment_five_crop
import h5py
import numpy as np
from torch.utils.data import Dataset
import torch
import gc
import h5py
import random
import numpy as np
from PIL import Image
from pathlib import Path

from torch.utils.data import Dataset
from torchvision.transforms import transforms
from super_image import Trainer, TrainingArguments, MsrnModel, MsrnConfig




train_dataset = TrainAugmentDatasetH5("/home/sofiahernandezgelado/Documents/MSRN/noise/train.h5", 1, patch_size = 32)                                                     # prepare the train dataset for loading PyTorch DataLoader
eval_dataset = EvalDatasetH5("/home/sofiahernandezgelado/Documents/MSRN/noise/eval.h5")      # prepare the eval dataset for the PyTorch DataLoader


training_args = TrainingArguments(
    output_dir='/home/sofiahernandezgelado/Documents/MSRN/noise/results',                 # output directory
    num_train_epochs= 300                  # total number of training epochs
)

config = MsrnConfig(
        scale=1,                                # train a model to upscale 4x
        bam=True,                                   # use balanced attention
    )

model = MsrnModel(config)

trainer = Trainer(
    model=model,                         # the instantiated model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=eval_dataset            # evaluation dataset
)

trainer.train()


!python /home/sofiahernandezgelado/Documents/MSRN/prepare.py\
    --images-dir "/home/sofiahernandezgelado/Documents/SRCNN_pytorch2/LargeTrial/train"\
    --output-path "/home/sofiahernandezgelado/Documents/SRCNN_pytorch2/LargeTrial/train_file.h5"\
    --scale 4


!python /home/sofiahernandezgelado/Documents/MSRN/prepare.py\
--images-dir "/home/sofiahernandezgelado/Documents/SRCNN_pytorch2/LargeTrial/eval"\
--output-path "/home/sofiahernandezgelado/Documents/SRCNN_pytorch2/LargeTrial/eval_file.h5"\
--scale 4 --eval


train_dataset = TrainAugmentDatasetH5("/home/sofiahernandezgelado/Documents/SRCNN_pytorch2/LargeTrial/train_file.h5", 4, patch_size = 32)                                                     # prepare the train dataset for loading PyTorch DataLoader
eval_dataset = EvalDatasetH5("/home/sofiahernandezgelado/Documents/SRCNN_pytorch2/LargeTrial/eval_file.h5")      # prepare the eval dataset for the PyTorch DataLoader


training_args = TrainingArguments(
    output_dir='/home/sofiahernandezgelado/Documents/MSRN/results',                 # output directory
    num_train_epochs= 300)

config = MsrnConfig(
    scale=4,                                # train a model to upscale 4x
    bam=True,                               # apply balanced attention to the network
)
model = MsrnModel(config)

trainer = Trainer(
    model=model,                         # the instantiated model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=eval_dataset            # evaluation dataset
)

trainer.train()

torch.cuda.empty_cache()

!python /home/sofiahernandezgelado/Documents/MSRN/prepare.py\
    --images-dir "/home/sofiahernandezgelado/Documents/SRCNN_pytorch2/LargeTrial/train"\
    --output-path "/home/sofiahernandezgelado/Documents/SRCNN_pytorch2/LargeTrial/train_file.h5"\
    --scale 8


!python /home/sofiahernandezgelado/Documents/MSRN/prepare.py\
--images-dir "/home/sofiahernandezgelado/Documents/SRCNN_pytorch2/LargeTrial/eval"\
--output-path "/home/sofiahernandezgelado/Documents/SRCNN_pytorch2/LargeTrial/eval_file.h5"\
--scale 8 --eval



train_dataset = TrainAugmentDatasetH5("/home/sofiahernandezgelado/Documents/SRCNN_pytorch2/LargeTrial/train_file.h5", 8, patch_size = 32)                                                     # prepare the train dataset for loading PyTorch DataLoader
eval_dataset = EvalDatasetH5("/home/sofiahernandezgelado/Documents/SRCNN_pytorch2/LargeTrial/eval_file.h5")      # prepare the eval dataset for the PyTorch DataLoader


training_args = TrainingArguments(
    output_dir='/home/sofiahernandezgelado/Documents/MSRN/results',                 # output directory
    num_train_epochs= 400,                  # total number of training epochs
)

config = MsrnConfig(
    scale=8,                                # train a model to upscale 4x
    bam=True,                               # apply balanced attention to the network
)
model = MsrnModel(config)

trainer = Trainer(
    model=model,                         # the instantiated model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=eval_dataset            # evaluation dataset
)

trainer.train()
