import os

# get current working directory
working_dir = os.getcwd()
os.chdir(working_dir + "/MSRN")

#import the necessary packages from super_image
from super_image.data import EvalDatasetH5, TrainAugmentDatasetH5, augment_five_crop
import numpy as np
from torch.utils.data import Dataset
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from super_image import Trainer, TrainingArguments, MsrnModel, MsrnConfig

os.chdir(working_dir)


# Load the training and eval dataset
train_dataset = TrainAugmentDatasetH5("data/train/train_file_MSRN.h5", 4, patch_size = 32)                                                     # prepare the train dataset for loading PyTorch DataLoader
eval_dataset = EvalDatasetH5("data/eval/eval_file_MSRN.h5")     


# Set the training arguments
training_args = TrainingArguments(
    output_dir='MSRN/results',              
    num_train_epochs= 400)

# Add BAM to the model, set upsampling scale
config = MsrnConfig(
    scale=4,                               
    bam=True,                               
)
model = MsrnModel(config)

# Trainer configuration
trainer = Trainer(
    model=model,                        
    args=training_args,                 
    train_dataset=train_dataset,         
    eval_dataset=eval_dataset          
)

# Train the model
trainer.train()

torch.cuda.empty_cache()
