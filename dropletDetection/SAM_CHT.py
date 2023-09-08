#Circle detection with SAM+CHT
import csv
import numpy as np
import os
import cv2
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from skimage.feature import peak_local_max, canny
import matplotlib.pyplot as plt
from skimage.transform import hough_circle, hough_circle_peaks
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch
import gc
import cv2
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-folder', type=str, required=True)  # Directory containing images
    parser.add_argument('--depthUm', type = int, required = True)
    parser.add_argument('--depthPx', type = int, required = True)
    parser.add_argument('--gpu', action='store_true', default=False)
    args = parser.parse_args()
    
    pattern = "x{}.tif"
    
    factor = args.depthUm/args.depthPx
    


    #get current working directory
    cwd = os.getcwd()
    # Load the SAM model and set the device
    sam = sam_model_registry["vit_h"](checkpoint= cwd + "/dropletDetection/sam_vit_h_4b8939.pth")
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # send to gpu if available
    if args.gpu:
        sam.to(device=DEVICE)
    #sam.to(device=DEVICE)

    # Create the mask generator
    mask_generator = SamAutomaticMaskGenerator(sam)


    # Create a log file to save the print statements
    csv_file_images = os.path.join(args.image_folder, "SAM_segmentation_metrics.csv")
    csvwriter_images = open(csv_file_images, 'w', newline='')
    csvwriter_images = csv.writer(csvwriter_images)
    
   # Write the header row
    csvwriter_images.writerow([
        "Image",
        "Diameter",
        "Average Diameter",
        "Number bubbles"
    ])
    

    # Iterate over the images in the folder
    for filename in os.listdir(args.image_folder):
        if filename.endswith(".tif"):
            # High-resolution image path
            high_res_image_path = os.path.join(args.image_folder, filename)

            # Load the images
            high_res_image = cv2.imread(high_res_image_path)

            # Generate masks for each image
            high_res_masks = mask_generator.generate(high_res_image)
          
            masks = [high_res_masks]
            combined_mask = np.zeros_like(high_res_masks[0]['segmentation'])
            all_combined = []
            for mask in masks:
                combined_mask = np.zeros_like(high_res_masks[0]['segmentation'])
                for segmentation in mask:
                    if 1000 < segmentation["area"] < 90000:
                        segmentation["label"] = "bubble"
                        combined_mask = np.logical_or(combined_mask, segmentation['segmentation'])
                    else:
                        segmentation["label"] = "background"
                all_combined.append(combined_mask)

            non_background_high_res = all_combined[0]
       
            edges_gt = canny(non_background_high_res, sigma=1)
         
             # Detect circles in ground truth
            hough_radii = np.arange(50, 180, 2)
            hough_res_gt = hough_circle(edges_gt, hough_radii)
            _, cx_gt, cy_gt, radii_gt = hough_circle_peaks(hough_res_gt, hough_radii, total_num_peaks=4, min_xdistance=250, min_ydistance=300, threshold =0.3 * np.max(hough_res_gt))

            number_gt = radii_gt.size
    
            binary_gt = np.zeros((high_res_image.shape[0], high_res_image.shape[1]), dtype=bool)
           
            for x_gt, y_gt, r_gt in zip(cx_gt, cy_gt, radii_gt):
                yy, xx = np.ogrid[:high_res_image.shape[0], :high_res_image.shape[1]]
                circle_mask = (xx - x_gt) ** 2 + (yy - y_gt) ** 2 <= r_gt ** 2
                binary_gt[circle_mask] = True
            
             
            # Create the masked images
            # boolean indexing and assignment based on mask
            color_img = np.zeros(high_res_image.shape).astype('uint8')
            
            red_mask_high_res = high_res_image.copy()
            red_mask_high_res[binary_gt] = [0, 0, 255]
 
            # Blend the masked image with the original image
            alpha = 0.4  # Opacity of the red mask (adjust as needed)
            red_mask_high_res = cv2.addWeighted(high_res_image, 1 - alpha, red_mask_high_res, alpha, 0)
         
            # Save the masked images
            masked_high_res_path = os.path.join(args.image_folder, f"SAM+CHT_{filename}")
           
            cv2.imwrite(masked_high_res_path, red_mask_high_res)


            # Write the data for each image
            csvwriter_images.writerow([
                filename,
                2 * radii_gt * factor,
                np.mean(2 * radii_gt * factor),
                number_gt
            ])
         
    

