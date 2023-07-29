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


def evaluate_circle_detection(args):
    image_folder = args.image_folder
    tolerance = 20
    
    bicubic_name = "_bicubic_x" + str(args.scale) + ".tif"
    srcnn_name = "_srcnn_x" + str(args.scale) + ".tif"
    #srcnn_name = "_srcnn_x" + str(args.scale) + ".tif"
    image_files = os.listdir(image_folder)
    gt_images = [file for file in image_files if not file.endswith(bicubic_name) and not file.endswith(srcnn_name) and not file.endswith('.txt')]
    #gt_images = [file for file in image_files if not file.endswith(bicubic_name) and not file.endswith(srcnn_name) if file.startswith("masked") and not file.endswith('.txt')]
    
    print(gt_images)

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
        for gt_image_name in gt_images:
            gt_image_path = os.path.join(image_folder, gt_image_name)
            bicubic_image_path = os.path.join(image_folder, gt_image_name.replace('.tif', bicubic_name))
            srcnn_image_path = os.path.join(image_folder, gt_image_name.replace('.tif', srcnn_name))
    
            gt_image = np.asarray(Image.open(gt_image_path).convert("L"))
            bicubic_image = np.asarray(Image.open(bicubic_image_path).convert("L"))
            srcnn_image = np.asarray(Image.open(srcnn_image_path).convert("L"))
            
            # Check if image sizes don't match
            #if gt_image.shape != bicubic_image.shape or gt_image.shape != srcnn_image.shape:
                # Rescale the ground truth image to match the sizes of bicubic and SRCNN images
             #   gt_image = resize(gt_image, bicubic_image.shape, preserve_range=True, anti_aliasing=True).astype(bicubic_image.dtype)

    
            edges_gt = canny(gt_image, sigma=1, low_threshold=10, high_threshold=13)
            edges_bicubic = canny(bicubic_image, sigma=1, low_threshold=10, high_threshold=13)
            edges_srcnn = canny(srcnn_image, sigma=1, low_threshold=10, high_threshold=13)
            
            #10, 13
            #50 100
    
            #85
            
            #plt.imshow(edges_gt, cmap = 'gray', vmin = 0, vmax = 1)
            #plt.axis('off')

            # Save the figure
            #output_path = os.path.join(image_folder, f"edges_{gt_image_name.replace('.tif', '.png')}")
            #plt.savefig(output_path, dpi=300)
            #plt.close()
            
            # Detect circles in ground truth
            hough_radii = np.arange(50, 180, 2)
            hough_res_gt = hough_circle(edges_gt, hough_radii)
            _, cx_gt, cy_gt, radii_gt = hough_circle_peaks(hough_res_gt, hough_radii, total_num_peaks=4, min_xdistance=250, min_ydistance=300, threshold =0.7 * np.max(hough_res_gt))
            total_gt_circles += len(cx_gt)
            #0.7
    
            # Detect circles in bicubic image
            hough_res_bicubic = hough_circle(edges_bicubic, hough_radii)
            _, cx_bicubic, cy_bicubic, radii_bicubic = hough_circle_peaks(hough_res_bicubic, hough_radii, total_num_peaks=4, min_xdistance=250, min_ydistance=300, threshold = 0.7 * np.max(hough_res_bicubic))
            detected_bicubic = len(cx_bicubic)
            total_detected_bicubic += detected_bicubic
    
            # Detect circles in SRCNN image
            hough_res_srcnn = hough_circle(edges_srcnn, hough_radii)
            _, cx_srcnn, cy_srcnn, radii_srcnn = hough_circle_peaks(hough_res_srcnn, hough_radii, total_num_peaks=4, min_xdistance=250, min_ydistance=300, threshold= 0.7 * np.max(hough_res_srcnn))
            detected_srcnn = len(cx_srcnn)
            total_detected_srcnn += detected_srcnn
            
            ## calculate the difference in radii
            # Calculate the absolute and relative differences in diameter
            diameter_gt = 2 * np.mean(radii_gt)
            diameter_bicubic = 2 * np.mean(radii_bicubic)
            diameter_srcnn = 2 * np.mean(radii_srcnn)
            
            absolute_diff_bicubic = np.abs(diameter_gt - diameter_bicubic)
            absolute_diff_srcnn = np.abs(diameter_gt - diameter_srcnn)
            
            relative_diff_bicubic = absolute_diff_bicubic / diameter_gt
            relative_diff_srcnn = absolute_diff_srcnn / diameter_gt
          
            srcnn_difference.append(absolute_diff_srcnn)
            bicubic_difference.append(absolute_diff_bicubic)
            srcnn_difference_relative.append(relative_diff_srcnn)
            bicubic_difference_relative.append(relative_diff_bicubic)
            
            
                
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
            
                  
            
            #print(f"Metrics for image: {gt_image_name}")
            
            # Calculate intersection and union for bicubic
            intersection = np.logical_and(binary_gt, binary_bicubic)
            union = np.logical_or(binary_gt, binary_bicubic)
            intersection_pixels = np.count_nonzero(intersection)
            union_pixels = np.count_nonzero(union)
            dsc_bicubic = (2.0 * intersection_pixels) / (union_pixels + intersection_pixels)
            dsc_bicubic_list.append(dsc_bicubic)
            
            # Calculate intersection and union for SRCNN
            intersection = np.logical_and(binary_gt, binary_srcnn)
            union = np.logical_or(binary_gt, binary_srcnn)
            intersection_pixels = np.count_nonzero(intersection)
            union_pixels = np.count_nonzero(union)
            dsc_srcnn = (2.0 * intersection_pixels) / (union_pixels + intersection_pixels)
            dsc_srcnn_list.append(dsc_srcnn)
            
            #print(f"Dice Similarity Coefficient (Bicubic): {dsc_bicubic}")
            #print(f"Dice Similarity Coefficient (SRCNN): {dsc_srcnn}")
            
            
            # Calculate pixel accuracy
            pixel_accuracy_bicubic = np.sum(binary_bicubic == binary_gt) / binary_gt.size
            pixel_accuracy_bicubic_list.append(pixel_accuracy_bicubic)
            pixel_accuracy_srcnn = np.sum(binary_srcnn == binary_gt) / binary_gt.size
            pixel_accuracy_srcnn_list.append(pixel_accuracy_srcnn)
            
            # Calculate intersection over union (IoU)
            intersection_over_union_bicubic = np.sum(np.logical_and(binary_bicubic, binary_gt)) / np.sum(np.logical_or(binary_bicubic, binary_gt))
            intersection_over_union_srcnn = np.sum(np.logical_and(binary_srcnn, binary_gt)) / np.sum(np.logical_or(binary_srcnn, binary_gt))
            iou_bicubic_list.append(intersection_over_union_bicubic)
            iou_srcnn_list.append(intersection_over_union_srcnn)
            
            #print(f"Pixel Accuracy (Bicubic): {pixel_accuracy_bicubic}")
            #print(f"Pixel Accuracy (SRCNN): {pixel_accuracy_srcnn}")
            #print(f"Intersection over Union (Bicubic): {intersection_over_union_bicubic}")
            #print(f"Intersection over Union (SRCNN): {intersection_over_union_srcnn}")
            
            # Print and write the metrics for the current image
            f.write("-------------------------------------------------------------------------")
            f.write("\n")
            f.write(f"Metrics for image: {gt_image_name}\n")
            f.write("\n")
            f.write(f"Dice Similarity Coefficient (Bicubic): {dsc_bicubic}\n")
            f.write(f"Dice Similarity Coefficient (SRCNN): {dsc_srcnn}\n")
            f.write(f"Pixel Accuracy (Bicubic): {pixel_accuracy_bicubic}\n")
            f.write(f"Pixel Accuracy (SRCNN): {pixel_accuracy_srcnn}\n")
            f.write(f"Intersection over Union (Bicubic): {intersection_over_union_bicubic}\n")
            f.write(f"Intersection over Union (SRCNN): {intersection_over_union_srcnn}\n")
            f.write("\n")
            
              
            # Print the results
            f.write(f"Absolute Difference - Bicubic: {absolute_diff_bicubic}\n")
            f.write(f"Absolute Difference - SRCNN: {absolute_diff_srcnn}\n")
            
            f.write(f"Relative Difference - Bicubic: {relative_diff_bicubic}\n")
            f.write(f"Relative Difference - SRCNN: {relative_diff_srcnn}\n")
            f.write("\n")
            
           
            # total_matched_bicubic += matched_bicubic
            # total_matched_srcnn += matched_srcnn
    
            #  # Print metrics for individual images
            # precision_bicubic, recall_bicubic, f1_score_bicubic = calculate_metrics(matched_bicubic, detected_bicubic, len(cx_gt))
            # precision_srcnn, recall_srcnn, f1_score_srcnn = calculate_metrics(matched_srcnn, detected_srcnn, len(cx_gt))
    
            # print(f"Metrics for image: {gt_image_name}")
            # print(f"Bicubic - Precision: {precision_bicubic}, Recall: {recall_bicubic}, F1 Score: {f1_score_bicubic}")
            # print(f"SRCNN - Precision: {precision_srcnn}, Recall: {recall_srcnn}, F1 Score: {f1_score_srcnn}")
            # print("")
            fontSize = 16
            font = font_manager.FontProperties(family='Times New Roman', size=fontSize)
    
                    # Draw circles on images
            fig, ax = plt.subplots(nrows=3, figsize=(6, 7))
            images = [gt_image, bicubic_image, srcnn_image]
            #images = [gt_image]
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
    
          # Calculate overall metrics for the entire test
        #overall_precision_bicubic, overall_recall_bicubic, overall_f1_score_bicubic = calculate_metrics(total_matched_bicubic, total_detected_bicubic, total_gt_circles)
       # overall_precision_srcnn, overall_recall_srcnn, overall_f1_score_srcnn = calculate_metrics(total_matched_srcnn, total_detected_srcnn, total_gt_circles)
    
        print("Overall Metrics")
        #print(f"Bicubic - Precision: {overall_precision_bicubic}, Recall: {overall_recall_bicubic}")
        #print(f"SRCNN - Precision: {overall_precision_srcnn}, Recall: {overall_recall_srcnn}")
        mean_dsc_bicubic = np.mean(dsc_bicubic_list)
        std_dsc_bicubic = np.std(dsc_bicubic_list)
        mean_pixel_accuracy_bicubic = np.mean(pixel_accuracy_bicubic_list)
        std_pixel_accuracy_bicubic = np.std(pixel_accuracy_bicubic_list)
        mean_iou_bicubic = np.mean(iou_bicubic_list)
        std_iou_bicubic = np.std(iou_bicubic_list)
        
        # Calculate overall metrics for SRCNN
        mean_dsc_srcnn = np.mean(dsc_srcnn_list)
        std_dsc_srcnn = np.std(dsc_srcnn_list)
        mean_pixel_accuracy_srcnn = np.mean(pixel_accuracy_srcnn_list)
        std_pixel_accuracy_srcnn = np.std(pixel_accuracy_srcnn_list)
        mean_iou_srcnn = np.mean(iou_srcnn_list)
        std_iou_srcnn = np.std(iou_srcnn_list)
        
        mean_srcnn_error = np.mean(srcnn_difference)
        std_srcnn_error = np.std(srcnn_difference)
        
        mean_bicubic_error = np.mean(bicubic_difference)
        std_bicubic_error = np.std(bicubic_difference)
        
        mean_srcnn_error_relative = np.mean(srcnn_difference_relative)
        std_srcnn_error_relative = np.std(srcnn_difference_relative)
        
        mean_bicubic_error_relative = np.mean(bicubic_difference_relative)
        std_bicubic_error_relative = np.std(bicubic_difference_relative)
        

        
        print(f"Bicubic - Mean DSC: {mean_dsc_bicubic}, Std DSC: {std_dsc_bicubic}")
        print(f"Bicubic - Mean Pixel Accuracy: {mean_pixel_accuracy_bicubic}, Std Pixel Accuracy: {std_pixel_accuracy_bicubic}")
        print(f"Bicubic - Mean IoU: {mean_iou_bicubic}, Std IoU: {std_iou_bicubic}")
        print("")
        print(f"SRGAN - Mean DSC: {mean_dsc_srcnn}, Std DSC: {std_dsc_srcnn}")
        print(f"SRGAN - Mean Pixel Accuracy: {mean_pixel_accuracy_srcnn}, Std Pixel Accuracy: {std_pixel_accuracy_srcnn}")
        print(f"SRGAN - Mean IoU: {mean_iou_srcnn}, Std IoU: {std_iou_srcnn}")
        
        # Print and write the overall metrics
        f.write("Overall Metrics\n")
        f.write(f"Bicubic - Mean DSC: {mean_dsc_bicubic}, Std DSC: {std_dsc_bicubic}\n")
        f.write(f"Bicubic - Mean Pixel Accuracy: {mean_pixel_accuracy_bicubic}, Std Pixel Accuracy: {std_pixel_accuracy_bicubic}\n")
        f.write(f"Bicubic - Mean IoU: {mean_iou_bicubic}, Std IoU: {std_iou_bicubic}\n")
        f.write("\n")
        f.write(f"SRGAN - Mean DSC: {mean_dsc_srcnn}, Std DSC: {std_dsc_srcnn}\n")
        f.write(f"SRGAN - Mean Pixel Accuracy: {mean_pixel_accuracy_srcnn}, Std Pixel Accuracy: {std_pixel_accuracy_srcnn}\n")
        f.write(f"SRGAN - Mean IoU: {mean_iou_srcnn}, Std IoU: {std_iou_srcnn}\n")
        f.write("\n")
        # Print the results
        f.write(f"Absolute Difference - Bicubic: {mean_bicubic_error}, Std: {std_bicubic_error}\n")
        f.write(f"Absolute Difference - SRCNN: {mean_srcnn_error}, Std: {std_srcnn_error}\n")
        
        f.write(f"Relative Difference - Bicubic: {mean_bicubic_error_relative}, Std: {std_bicubic_error_relative}\n")
        f.write(f"Relative Difference - SRCNN: {mean_srcnn_error_relative}, Std: {std_srcnn_error_relative}\n")
        f.write("\n")
        f.write("-------------------------------------------------------------------------")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-folder', type=str, required=True, help='Path to test images directory')
    parser.add_argument('--scale', type=int, required=True, help='Scale')
    parser.add_argument('--rgb', type = bool, default = False)
    args = parser.parse_args()
    
    evaluate_circle_detection(args)

