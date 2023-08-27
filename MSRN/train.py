import os
os.chdir("/home/SuperResolution_Microfluidics/MSRN")

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

os.chdir("/home/SuperResolution_Microfluidics")

train_dataset = TrainAugmentDatasetH5("data/train/train_file_MSRN.h5", 4, patch_size = 32)                                                     # prepare the train dataset for loading PyTorch DataLoader
eval_dataset = EvalDatasetH5("data/eval/eval_file_MSRN.h5")      # prepare the eval dataset for the PyTorch DataLoader


training_args = TrainingArguments(
    output_dir='MSRN/results',                 # output directory
    num_train_epochs= 400)

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
