#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 09:10:38 2023

@author: daniel
"""

import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import os
from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, ssim


def calc_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return psnr

def calc_ssim(image1, image2):
    ssim_score = ssim(image1, image2)
    return ssim_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-dir', type=str, required=True)  # Directory containing images
    parser.add_argument('--scale', type=int, default=3)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = SRCNN().to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    ssim_total_pred = 0.0
    ssim_total_bicubic = 0.0
    psnr_total_pred = 0.0
    psnr_total_bicubic = 0.0
    num_images = 0
    
    ssim_scores_pred = []
    ssim_scores_bicubic = []
    psnr_scores_pred = []
    psnr_scores_bicubic = []
    
    bicubic_name = "_bicubic_x" + str(args.scale) + ".tif"
    srcnn_name = "_srcnn_x" + str(args.scale) + ".tif"

    imfiles = os.listdir(args.image_dir)
    image_files = [file for file in imfiles if not file.endswith(bicubic_name) and not file.endswith(srcnn_name) and not file.endswith('.txt') and not file.endswith('.png')]
    
    
    output_path = os.path.join(args.image_dir, "test_metrics.txt")


    # Open the file for writing
    with open(output_path, 'w') as f:
    
        for image_file in image_files:
            image_path = os.path.join(args.image_dir, image_file)
    
            img = pil_image.open(image_path).convert('L')
            hr_width = (img.width // args.scale) * args.scale
            hr_height = (img.height // args.scale) * args.scale
            image_hr = img.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
            lr = image_hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
            image_lr = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
            
            
            image_lr.save(image_path.replace('.', '_bicubic_x{}.'.format(args.scale)))
            image_hr.save(image_path)
    
            image = np.array(image_hr).astype(np.float32)
           
    
            imagelr = np.array(image_lr).astype(np.float32)
            imagelr_norm = imagelr/ 255  # Normalize the pixel values to the range of [0, 1]
    
            y = torch.from_numpy(imagelr_norm).to(device)
            y = y.unsqueeze(0).unsqueeze(0)
    
            with torch.no_grad():
                preds = model(y).clamp(0.0, 1.0)
                
            preds = preds.mul(255).cpu().numpy().squeeze(0).squeeze(0)
            output = np.clip(preds, 0.0, 255).astype(np.uint8)
            output_img = pil_image.fromarray(output)
            output_img.save(image_path.replace('.', '_srcnn_x{}.'.format(args.scale)))
    
            # Calculate SSIM between predicted high-resolution and ground truth high-resolution images
            ssim_pred = calc_ssim(image, output)
            ssim_total_pred += ssim_pred
            ssim_scores_pred.append(ssim_pred)
    
            # Calculate SSIM between ground truth high-resolution and low-resolution images
            ssim_bicubic = calc_ssim(image, imagelr)
            ssim_total_bicubic += ssim_bicubic
            ssim_scores_bicubic.append(ssim_bicubic)
    
            # Calculate PSNR between predicted high-resolution and ground truth high-resolution images
            psnr_pred = calc_psnr(image, output)
            psnr_total_pred += psnr_pred
            psnr_scores_pred.append(psnr_pred)
    
            # Calculate PSNR between ground truth high-resolution and low-resolution images
            psnr_bicubic = calc_psnr(image, imagelr)
            psnr_total_bicubic += psnr_bicubic
            psnr_scores_bicubic.append(psnr_bicubic)
    
            num_images += 1
    
        
    
        # Calculate average SSIM and PSNR
        ssim_avg_pred = ssim_total_pred / num_images
        ssim_avg_bicubic = ssim_total_bicubic / num_images
        psnr_avg_pred = psnr_total_pred / num_images
        psnr_avg_bicubic = psnr_total_bicubic / num_images
    
        # Calculate standard deviations for SSIM and PSNR
        ssim_std_pred = np.std(ssim_scores_pred)
        ssim_std_bicubic = np.std(ssim_scores_bicubic)
        psnr_std_pred = np.std(psnr_scores_pred)
        psnr_std_bicubic = np.std(psnr_scores_bicubic)
        
        # Calculate increase in SSIM and PSNR
        ssim_increase = ssim_avg_pred - ssim_avg_bicubic
        psnr_increase = psnr_avg_pred - psnr_avg_bicubic
        
        f.write(f'Average SSIM (prediction): {ssim_avg_pred}\n')
        f.write(f'Standard Deviation SSIM (prediction): {ssim_std_pred}\n')
        f.write(f'Average SSIM (bicubic): {ssim_avg_bicubic}\n')
        f.write(f'Standard Deviation SSIM (bicubic): {ssim_std_bicubic}\n')
        f.write(f'Average PSNR (prediction): {psnr_avg_pred}\n')
        f.write(f'Standard Deviation PSNR (prediction): {psnr_std_pred}\n')
        f.write(f'Average PSNR (bicubic): {psnr_avg_bicubic}\n')
        f.write(f'Standard Deviation PSNR (bicubic): {psnr_std_bicubic}\n')
        f.write(f'Increase in SSIM: {ssim_increase}\n')
        f.write(f'Increase in PSNR: {psnr_increase}\n')
