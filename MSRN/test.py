#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 17:13:41 2023

@author: sofiahernandezgelado
"""

## Run inference for MSRN
import os
#get current working directory
working_dir = os.getcwd()
os.chdir(working_dir + "/MSRN")
checkpoint_path = 'MSRN/results'
scale = 4
from super_image import MsrnModel
from PIL import Image
import torch
from skimage.metrics import structural_similarity as ssim
import numpy as np
import numpy as np
import cv2
from PIL import Image

import torch
from torch import Tensor
from torchvision.transforms.functional import rgb_to_grayscale
import cv2
from PIL import Image
import numpy as np
import torch
from torch import Tensor

os.chdir(working_dir)

class ImageLoader:
    @staticmethod
    def load_image(image: Image):
        lr = np.array(image.convert('RGB'))
        lr = lr[::].astype(np.float32).transpose([2, 0, 1]) / 255.0
        return torch.as_tensor([lr])

    @staticmethod
    def _process_image_to_save(pred: Tensor):
        pred = pred.data.cpu().numpy()
        pred = pred[0].transpose((1, 2, 0)) * 255.0
        pred_gray = cv2.cvtColor(pred.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        return pred_gray
    
    @staticmethod
    def save_image(pred: Tensor, output_file: str):
        pred = ImageLoader._process_image_to_save(pred)
        image = Image.fromarray(pred)
        image.save(output_file, format='TIFF')
    
    @staticmethod
    def save_compare(input: Tensor, pred: Tensor, output_file: str):
        pred = ImageLoader._process_image_to_save(pred)
        input = ImageLoader._process_image_to_save(input)
        input_resize = cv2.resize(input, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_CUBIC)
        hstack = np.hstack((input_resize, pred))
        image = Image.fromarray(hstack)
        image.save(output_file, format='TIFF')
    
    
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return psnr

def calculate_ssim(image1, image2):
    # Convert tensors to numpy arrays and adjust dimensions if necessary

    # Calculate SSIM
    ssim_score = ssim(image1, image2, data_range = image2.max() - image2.min(), channel_axis=1)
    return ssim_score


def upscale_image(model, image_path, scale):
    hr = Image.open(image_path).convert('RGB')
    hr_width = (hr.width // scale) * scale
    hr_height = (hr.height // scale) * scale
    hr = hr.resize((hr_width, hr_height), resample=Image.BICUBIC)
    lr = hr.resize((hr_width // scale, hr_height // scale), resample=Image.BICUBIC)
    inputs = ImageLoader.load_image(lr)
    preds = model(inputs)
    return preds

# Directory containing test images
# Directory containing test images
input_dir = 'data/test/'
scale = 4
checkpoint_path = 'MSRN/checkpoints/'
hr_dir = 'data/test/'

# Load the MSRN model
model = MsrnModel.from_pretrained(checkpoint_path, scale=scale)  # Specify the path to your MSRN model

all_psnr_bicubic =[]
all_ssim_bicubic = []
all_psnr_msrn =[]
all_ssim_msrn = []

# Upscale images and calculate PSNR and SSIM
for filename in os.listdir(input_dir):
    if filename.endswith(".tif"):
        image_path = os.path.join(input_dir, filename)
        lr = Image.open(image_path)
        image_lr = lr.resize((lr.width//scale, lr.height//scale), resample=Image.BICUBIC)
        lr = image_lr.resize((image_lr.width * scale, image_lr.height * scale), resample=Image.BICUBIC)
        lr.save(image_path.replace('.', '_bicubic_x{}.'.format(scale)))
        
        inputs = ImageLoader.load_image(image_lr)
        preds = model(inputs)
        ImageLoader.save_image(preds, image_path.replace('.', '_msrn_x{}.'.format(scale))) 
 
        hr_image = Image.open(hr_dir + filename).convert('L')
        hr_image = hr_image.resize((lr.width, lr.height), resample=Image.BICUBIC)
        lr_image = Image.open(image_path.replace('.', '_bicubic_x{}.'.format(scale))).convert('L')
        msrn_image = Image.open(image_path.replace('.', '_msrn_x{}.'.format(scale))).convert('L')

        psnr_bicubic = calculate_psnr(np.asarray(hr_image), np.asarray(lr_image))
        psnr_msrn = calculate_psnr(np.asarray(hr_image), np.asarray(msrn_image))
        all_psnr_bicubic.append(psnr_bicubic)
        all_psnr_msrn.append(psnr_msrn)
        
        ssim_bicubic = calculate_ssim(np.asarray(hr_image), np.asarray(lr_image))
        ssim_msrn = calculate_ssim(np.asarray(hr_image), np.asarray(msrn_image))
        all_ssim_bicubic.append(ssim_bicubic)
        all_ssim_msrn.append(ssim_msrn)


#if does not exist create file


output_path = "MSRN/test/test_metrics.txt"
# Open the file for writing
with open(output_path, 'w') as f:
    f.write(f'Average PSNR (bicubic): {np.mean(all_psnr_bicubic)}, Standard Deviation PSNR (bicubic): {np.std(all_psnr_bicubic)}\n')
    f.write(f'Average PSNR (MSRN): {np.mean(all_psnr_msrn)}, Standard Deviation PSNR (MSRN): {np.std(all_psnr_msrn)}\n')
    f.write(f'Average SSIM (bicubic): {np.mean(all_ssim_bicubic)}, Standard Deviation SSIM (bicubic): {np.std(all_ssim_bicubic)}\n')
    f.write(f'Average SSIM (MSRN): {np.mean(all_ssim_msrn)}, Standard Deviation SSIM (MSRN): {np.std(all_ssim_msrn)}\n')