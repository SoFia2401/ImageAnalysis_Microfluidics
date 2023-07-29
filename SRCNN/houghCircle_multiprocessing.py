#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 11:16:18 2023

@author: sofiahernandezgelado
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 11:43:10 2023
@author: daniel
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import peak_local_max, canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from skimage.io import imread, imsave
from skimage.morphology import disk
from skimage.transform import resize
import argparse
import matplotlib.font_manager as font_manager
from skimage.color import rgb2gray
from PIL import Image
import multiprocessing
from functools import partial

# Function to draw a cross
def draw_cross(image, y, x, length, color, width):
    half_width = width // 2
    image[y - half_width: y + half_width, x - length: x + length] = color
    image[y - length: y + length, x - half_width: x + half_width] = color


def calculate_metrics(true_positives, detected, total_gt):
    precision = true_positives / detected if detected > 0 else 0
    recall = true_positives / total_gt if total_gt > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score


def distance(center1, center2):
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)


def process_image(image_folder, tolerance, bicubic_name, srcnn_name, gt_image_name):
    gt_image_path = os.path.join(image_folder, gt_image_name)
    bicubic_image_path = os.path.join(image_folder, gt_image_name.replace('.tif', bicubic_name))
    srcnn_image_path = os.path.join(image_folder, gt_image_name.replace('.tif', srcnn_name))

    gt_image = np.asarray(Image.open(gt_image_path).convert("L"))
    bicubic_image = np.asarray(Image.open(bicubic_image_path).convert("L"))
    srcnn_image = np.asarray(Image.open(srcnn_image_path).convert("L"))

    edges_gt = canny(gt_image, sigma=1, low_threshold=10, high_threshold=13)
    edges_bicubic = canny(bicubic_image, sigma=1, low_threshold=10, high_threshold=13)
    edges_srcnn = canny(srcnn_image, sigma=1, low_threshold=10, high_threshold=13)

    # Detect circles in ground truth
          # Detect circles in ground truth
    hough_radii = np.arange(50, 180, 2)
    hough_res_gt = hough_circle(edges_gt, hough_radii)
    _, cx_gt, cy_gt, radii_gt = hough_circle_peaks(hough_res_gt, hough_radii, total_num_peaks=4, min_xdistance=250, min_ydistance=300, threshold =0.7 * np.max(hough_res_gt))
    #0.7

    # Detect circles in bicubic image
    hough_res_bicubic = hough_circle(edges_bicubic, hough_radii)
    _, cx_bicubic, cy_bicubic, radii_bicubic = hough_circle_peaks(hough_res_bicubic, hough_radii, total_num_peaks=4, min_xdistance=250, min_ydistance=300, threshold = 0.7 * np.max(hough_res_bicubic))
    detected_bicubic = len(cx_bicubic)

    # Detect circles in SRCNN image
    hough_res_srcnn = hough_circle(edges_srcnn, hough_radii)
    _, cx_srcnn, cy_srcnn, radii_srcnn = hough_circle_peaks(hough_res_srcnn, hough_radii, total_num_peaks=4, min_xdistance=250, min_ydistance=300, threshold= 0.7 * np.max(hough_res_srcnn))
    detected_srcnn = len(cx_srcnn)

    ## calculate the difference in radii
    # Calculate the absolute and relative differences in diameter
    diameter_gt = 2 * np.mean(radii_gt)
    diameter_bicubic = 2 * np.mean(radii_bicubic)
    diameter_srcnn = 2 * np.mean(radii_srcnn)
    
    absolute_diff_bicubic = np.abs(diameter_gt - diameter_bicubic)
    absolute_diff_srcnn = np.abs(diameter_gt - diameter_srcnn)
    
    relative_diff_bicubic = absolute_diff_bicubic / diameter_gt
    relative_diff_srcnn = absolute_diff_srcnn / diameter_gt
  

    # Compare number of detected circles with ground truth
    matched_bicubic = sum(
        any(distance((x_gt, y_gt), (x_b, y_b)) <= tolerance for x_b, y_b in zip(cx_bicubic, cy_bicubic))
        for x_gt, y_gt in zip(cx_gt, cy_gt)
    )
    matched_srcnn = sum(
        any(distance((x_gt, y_gt), (x_s, y_s)) <= tolerance for x_s, y_s in zip(cx_srcnn, cy_srcnn))
        for x_gt, y_gt in zip(cx_gt, cy_gt)
    )
    
    # Create binary images for ground truth and detected circles
    # Create binary images for ground truth and detected circles
    binary_gt = np.zeros_like(gt_image, dtype=bool)
    binary_bicubic = np.zeros_like(bicubic_image, dtype=bool)
    binary_srcnn = np.zeros_like(srcnn_image, dtype=bool)
    
    for x_gt, y_gt, r_gt in zip(cx_gt, cy_gt, radii_gt):
        yy, xx = np.ogrid[:gt_image.shape[0], :gt_image.shape[1]]
        circle_mask = (xx - x_gt) ** 2 + (yy - y_gt) ** 2 <= r_gt ** 2
        binary_gt[circle_mask] = True
    
    for x_b, y_b, r_b in zip(cx_bicubic, cy_bicubic, radii_bicubic):
        yy, xx = np.ogrid[:bicubic_image.shape[0], :bicubic_image.shape[1]]
        circle_mask = (xx - x_b) ** 2 + (yy - y_b) ** 2 <= r_b ** 2
        binary_bicubic[circle_mask] = True
    
    for x_s, y_s, r_s in zip(cx_srcnn, cy_srcnn, radii_srcnn):
        yy, xx = np.ogrid[:srcnn_image.shape[0], :srcnn_image.shape[1]]
        circle_mask = (xx - x_s) ** 2 + (yy - y_s) ** 2 <= r_s ** 2
        binary_srcnn[circle_mask] = True
    
      
    # Calculate intersection and union for bicubic
    intersection = np.logical_and(binary_gt, binary_bicubic)
    union = np.logical_or(binary_gt, binary_bicubic)
    intersection_pixels = np.count_nonzero(intersection)
    union_pixels = np.count_nonzero(union)
    dsc_bicubic = (2.0 * intersection_pixels) / (union_pixels + intersection_pixels)

    # Calculate intersection and union for SRCNN
    intersection = np.logical_and(binary_gt, binary_srcnn)
    union = np.logical_or(binary_gt, binary_srcnn)
    intersection_pixels = np.count_nonzero(intersection)
    union_pixels = np.count_nonzero(union)
    dsc_srcnn = (2.0 * intersection_pixels) / (union_pixels + intersection_pixels)
    
    #print(f"Dice Similarity Coefficient (Bicubic): {dsc_bicubic}")
    #print(f"Dice Similarity Coefficient (SRCNN): {dsc_srcnn}")
    
    
    # Calculate pixel accuracy
    pixel_accuracy_bicubic = np.sum(binary_bicubic == binary_gt) / binary_gt.size
    pixel_accuracy_srcnn = np.sum(binary_srcnn == binary_gt) / binary_gt.size
    
    # Calculate intersection over union (IoU)
    intersection_over_union_bicubic = np.sum(np.logical_and(binary_bicubic, binary_gt)) / np.sum(np.logical_or(binary_bicubic, binary_gt))
    intersection_over_union_srcnn = np.sum(np.logical_and(binary_srcnn, binary_gt)) / np.sum(np.logical_or(binary_srcnn, binary_gt))

# Draw circles on images
    fontSize = 16
    font = font_manager.FontProperties(family='Times New Roman', size=fontSize)
 
    fig, ax = plt.subplots(nrows=3, figsize=(6, 7))
    images = [gt_image, bicubic_image, srcnn_image]
   # images = [gt_image]
    titles = ['Original', 'Bicubic', 'SRCNN']
    line_width = 1  # Adjust the line width as desired
   
    for i, image in enumerate(images):
        image_with_circles = np.copy(color.gray2rgb(image))
        if i == 0:
            cx = cx_gt
            cy = cy_gt
            radii = radii_gt
        elif i == 1:
            cx = cx_bicubic
            cy = cy_bicubic
            radii = radii_bicubic
        else:
            cx = cx_srcnn
            cy = cy_srcnn
            radii = radii_srcnn
   
        for center_y, center_x, radius in zip(cy, cx, radii):
            circy, circx = circle_perimeter(center_y, center_x, radius, shape=image.shape, method='andres')
            
                # Draw a thicker circle
            for w in range(-line_width, line_width):
 
                image_with_circles[circy, circx+w] = (220, 20, 20)
                    
                image_with_circles[circy+w, circx] = (220, 20, 20)
                    
   
 
           #image_with_circles[circy, circx] = (220, 20, 20)
           
           # Draw a cross at the center of the circle
            draw_cross(image_with_circles, center_y, center_x, length=8, color=(20, 220, 20), width=4)
        ax[i].set_title(titles[i], font=font)
        ax[i].imshow(image_with_circles)
        #plt.axis('off')
        ax[i].axis('off')
   
   # Save the figure
    output_path = os.path.join(image_folder, f"circles_{gt_image_name.replace('.tif', '.png')}")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    
    
        

    return gt_image_name, absolute_diff_srcnn, absolute_diff_bicubic, relative_diff_srcnn, relative_diff_bicubic, matched_srcnn, dsc_bicubic, dsc_srcnn, pixel_accuracy_srcnn, pixel_accuracy_bicubic, intersection_over_union_bicubic, intersection_over_union_srcnn


def evaluate_circle_detection(image_folder, scale, rgb):
    tolerance = 20
    bicubic_name = "_bicubic_x" + str(scale) + ".tif"
    srcnn_name = "_srcnn_x" + str(scale) + ".tif"
    image_files = os.listdir(image_folder)
    gt_images = [file for file in image_files if not file.endswith(bicubic_name) and not file.endswith(srcnn_name) and not file.endswith('.txt')]

    total_detected_bicubic = 0
    total_detected_srcnn = 0
    total_gt_circles = 0
    total_matched_bicubic = 0
    total_matched_srcnn = 0

    dsc_bicubic_list = []
    dsc_srcnn_list = []
    pixel_accuracy_bicubic_list = []
    pixel_accuracy_srcnn_list = []
    iou_bicubic_list = []
    iou_srcnn_list = []

    srcnn_difference = []
    bicubic_difference = []
    srcnn_difference_relative = []
    bicubic_difference_relative = []

    output_path = os.path.join(image_folder, "segmentation_metrics.txt")

    # Open the file for writing
    with open(output_path, 'w') as f:
        num_processes = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=num_processes)

        partial_func = partial(process_image, image_folder, tolerance, bicubic_name, srcnn_name)
        results = pool.map(partial_func, gt_images)

        for result in results:
            gt_image_name, absolute_diff_srcnn, absolute_diff_bicubic, relative_diff_srcnn, relative_diff_bicubic, matched_srcnn, dsc_bicubic, dsc_srcnn, pixel_accuracy_srcnn, pixel_accuracy_bicubic, intersection_over_union_bicubic, intersection_over_union_srcnn = result


            dsc_bicubic_list.append(dsc_bicubic)
            dsc_srcnn_list.append(dsc_srcnn)
            pixel_accuracy_bicubic_list.append(pixel_accuracy_bicubic)
            pixel_accuracy_srcnn_list.append(pixel_accuracy_srcnn)
            iou_bicubic_list.append(intersection_over_union_bicubic)
            iou_srcnn_list.append(intersection_over_union_srcnn)
            srcnn_difference.append(absolute_diff_srcnn)
            bicubic_difference.append(absolute_diff_bicubic)
            srcnn_difference_relative.append(relative_diff_srcnn)
            bicubic_difference_relative.append(relative_diff_bicubic)
            
            f.write("-------------------------------------------------------------------------")
            f.write("\n")
            f.write(f"Metrics for image: {gt_image_name}\n")
            f.write("\n")
            f.write(f"Bicubic - Dice Similarity Coefficient: {dsc_bicubic}\n")
            f.write(f"SRCNN - Dice Similarity Coefficient: {dsc_srcnn}\n")
            f.write(f"Bicubic - Pixel Accuracy: {pixel_accuracy_bicubic}\n")
            f.write(f"SRCNN - Pixel Accuracy: {pixel_accuracy_srcnn}\n")
            f.write(f"Bicubic - Intersection over Union: {intersection_over_union_bicubic}\n")
            f.write(f"SRCNN - Intersection over Union: {intersection_over_union_srcnn}\n")
            f.write(f"Bicubic - Absolute Difference: {absolute_diff_bicubic}\n")
            f.write(f"SRCNN - Absolute Difference: {absolute_diff_srcnn}\n")  
            f.write(f"Bicubic - Relative Difference: {relative_diff_bicubic}\n")
            f.write(f"SRCNN - Relative Difference: {relative_diff_srcnn}\n")
            f.write("\n")

        # Calculate overall metrics for the entire test
        # Print and write the overall metrics
        
        f.write("Overall Metrics\n")
        f.write("--------------------")
        f.write(f"Bicubic - Mean DSC: {np.mean(dsc_bicubic_list)}, Std DSC: {np.std(dsc_bicubic_list)}\n")
        f.write(f"Bicubic - Mean Pixel Accuracy: {np.mean(pixel_accuracy_bicubic_list)}, Std Pixel Accuracy: {np.std(pixel_accuracy_bicubic_list)}\n")
        f.write(f"Bicubic - Mean IoU: {np.mean(iou_bicubic_list)}, Std IoU: {np.std(iou_bicubic_list)}\n")
        f.write(f"Bicubic - Absolute Difference: {np.mean(bicubic_difference)}, Std: {np.std(bicubic_difference)}\n")
        
        f.write(f"Bicubic - Relative Difference: {np.mean(bicubic_difference_relative)}, Std: {np.std(bicubic_difference_relative)}\n")
        f.write("\n")
        f.write(f"SRCNN - Mean DSC: {np.mean(dsc_srcnn_list)}, Std DSC: {np.std(dsc_srcnn_list)}\n")
        
        f.write(f"SRCNN - Mean Pixel Accuracy: {np.mean(pixel_accuracy_srcnn_list)}, Std Pixel Accuracy: {np.std(pixel_accuracy_srcnn_list)}\n")
        f.write(f"SRCNN - Mean IoU: {np.mean(iou_srcnn_list)}, Std IoU: {np.std(iou_srcnn_list)}\n")
        f.write(f"SRCNN - Absolute Difference: {np.mean(srcnn_difference)}, Std: {np.std(srcnn_difference)}\n")
        f.write(f"SRCNN - Relative Difference: {np.mean(srcnn_difference_relative)}, Std: {np.std(srcnn_difference_relative)}\n")
        
        f.write("-------------------------------------------------------------------------")
        #print("got here 2")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-folder', type=str, required=True, help='Path to test images directory')
    parser.add_argument('--scale', type=int, required=True, help='Scale')
    parser.add_argument('--rgb', type=bool, default=False)
    args = parser.parse_args()

    evaluate_circle_detection(args.image_folder, args.scale, args.rgb)
