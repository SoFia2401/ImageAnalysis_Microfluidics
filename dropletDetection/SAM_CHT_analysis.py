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

# Define functions to calculate Dice score and IoU

def calculate_dice_score(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    dice_score = (2.0 * intersection.sum()) / (mask1.sum() + mask2.sum())
    return dice_score

def calculate_iou(mask1, mask2):
    union = np.logical_or(mask1, mask2)
    intersection = np.logical_and(mask1, mask2)
    iou = intersection.sum() / union.sum()
    return iou


def find_closest_circles(cx_gt, cy_gt, cx_pred, cy_pred, tolerance=10):
    """
    Find the closest circles based on their center points.
    
    Parameters:
        cx_gt (ndarray): Array of x-coordinates of ground truth circles.
        cy_gt (ndarray): Array of y-coordinates of ground truth circles.
        cx_pred (ndarray): Array of x-coordinates of predicted circles.
        cy_pred (ndarray): Array of y-coordinates of predicted circles.
        tolerance (int): Maximum distance (in pixels) between center points to consider circles as matching.

    Returns:
        matched_indices_gt (list): Indices of matched ground truth circles.
        matched_indices_pred (list): Indices of matched predicted circles.
    """
    matched_indices_gt = []
    matched_indices_pred = []
    
    for i in range(len(cx_gt)):
        match_index = -1
        
        for j in range(len(cx_pred)):
            distance = np.sqrt((cx_gt[i] - cx_pred[j])**2 + (cy_gt[i] - cy_pred[j])**2)
            if distance <= tolerance:
                match_index = j
        
        if match_index != -1:
            matched_indices_gt.append(i)
            matched_indices_pred.append(match_index)
    
    return matched_indices_gt, matched_indices_pred


def find_closest_circles_replace_for_0(cx_gt, cy_gt, cx_pred, cy_pred, radii_gt, radii_pred, factor, tolerance):
    """
    Find the closest circles based on their center points.
    
    Parameters:
        cx_gt (ndarray): Array of x-coordinates of ground truth circles.
        cy_gt (ndarray): Array of y-coordinates of ground truth circles.
        cx_pred (ndarray): Array of x-coordinates of predicted circles.
        cy_pred (ndarray): Array of y-coordinates of predicted circles.
        tolerance (int): Maximum distance (in pixels) between center points to consider circles as matching.

    Returns:
        matched_indices_gt (list): Indices of matched ground truth circles.
        matched_indices_pred (list): Indices of matched predicted circles.
    """
    
    matched_radii_pred = np.array([])
    matched_radii_gt = np.array([])
    
    for i in range(cx_gt.size):
        match_index = -1
        
        for j in range(cx_pred.size):
            distance = np.sqrt((cx_gt[i] - cx_pred[j])**2 + (cy_gt[i] - cy_pred[j])**2)
            if distance <= tolerance:
                match_index = j
        
        if match_index != -1:
            matched_radii_pred = np.append(matched_radii_pred, radii_pred[match_index] * factor * 2)
            matched_radii_gt = np.append(matched_radii_gt, radii_gt[i] * factor * 2)
        else:
            matched_radii_pred = np.append(matched_radii_pred, 0.0)
            matched_radii_gt = np.append(matched_radii_gt, radii_gt[i] * factor *2)
            
            
    return matched_radii_gt, matched_radii_pred

       


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, required=True)
    parser.add_argument('--image-folder', type=str, required=True)  # Directory containing images
    parser.add_argument('--model', type = str, required = True)
    parser.add_argument('--depthUm', type = int, required = True)
    parser.add_argument('--depthPx', type = int, required = True)
    parser.add_argument('--gpu', type = bool, default=False)
    args = parser.parse_args()
    
    pattern = "x{}.tif"
    
    factor = args.depthUm/args.depthPx
    


    #get current working directory
    cwd = os.getcwd()
    # Load the SAM model and set the device
    sam = sam_model_registry["vit_h"](checkpoint= cwd + "/dropletDetection/sam_vit_h_4b8939.pth")
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # send to gpu if available
    #if(args.gpu == True):
        #sam.to(device=DEVICE)
    #sam.to(device=DEVICE)

    # Create the mask generator
    mask_generator = SamAutomaticMaskGenerator(sam)

    # Create empty lists to store dice scores and IoU values
    dice_scores_bicubic = []
    dice_scores_model = []
    ious_bicubic = []
    ious_model = []
    model_difference_relative_list = []
    bicubic_differences_relative_list = []
    bicubic_detection_list = []
    model_detection_list = []
    number_bicubic_list = []
    number_model_list = []
    number_gt_list = []
    differences_model_zero_list = []
    differences_bicubic_zero_list = []

    # Create a log file to save the print statements
    csv_file_images = os.path.join(args.image_folder, "SAM_segmentation_metrics.csv")
    csvwriter_images = open(csv_file_images, 'w', newline='')
    csvwriter_images = csv.writer(csvwriter_images)
    
        # Write the header row
    csvwriter_images.writerow([
        "Image",
        f"Dice score (Bicubic)", f"Dice score ({args.model})",
        "IoU (Bicubic)", f"IoU ({args.model})",
        "Diameter HR", f"Diameter (Bicubic)", f"Diameter ({args.model})",
        "Average Diameter HR", f"Average Diameter (Bicubic)", f"Average Diameter ({args.model})",
        "Absolute Difference - Bicubic", f"Absolute Difference - {args.model}",
        "Relative Difference - Bicubic", f"Relative Difference - {args.model}",
        "Diameter zero filled HR", f"Diameter zero filled (Bicubic)", f"Diameter zero filled ({args.model})",
        "Absolute Difference zero filled - Bicubic", f"Absolute Difference zero filled - {args.model}",
        "Percentage Detection - Bicubic", f"Percentage Detection - {args.model}",
        "Number bubbles - Gt", "Number bubbles - Bicubic", f"Number bubbles model - {args.model}"
    ])
    
    csv_file_overall = os.path.join(args.image_folder, "SAM_segmentation_metrics_overall.csv")
    csvwriter_overall = open(csv_file_overall, 'w', newline='')
    csvwriter_overall = csv.writer(csvwriter_overall)
    
    # Write the column names for each metric
    csvwriter_overall.writerow([
        "Dice score (Bicubic)",
        "Std Dice score (Bicubic)",
        f"Dice score ({args.model}",
        f"Std Dice score ({args.model}",
        "IoU (Bicubic)",
        "Std IoU (Bicubic)",
        f"IoU ({args.model}", 
        f"Std IoU ({args.model}", 
        "Absolute Difference - Bicubic", 
        "Std Absolute Difference - Bicubic", 
        f"Absolute Difference - {args.model}", 
        f"Std Absolute Difference - {args.model}", 
        "Relative Difference - Bicubic", 
        "Std Relative Difference - Bicubic", 
        f"Relative Difference - {args.model}", 
        f"Std Relative Difference - {args.model}",
        "Absolute Difference zero filled - Bicubic",
        "Std Absolute Difference zero filled - Bicubic", 
        f"Absolute Difference zero filled- {args.model}",
        f"Std Absolute Difference zero filled- {args.model}", 
        "Percentage Detection - Bicubic:", 
        "Std Percentage Detection - Bicubic:", 
        f"Percentage Detection - {args.model}",
        f"Std Percentage Detection - {args.model}"
    ])

    


    model_difference = []
    bicubic_difference = []
    

    # Iterate over the images in the folder
    for filename in os.listdir(args.image_folder):
        if filename.endswith(".tif") and not filename.endswith(pattern.format(args.scale)) and not filename.startswith("hough") and not filename.startswith("SAM+CHT") and not filename.endswith(".log") and not filename.endswith(".csv"):
            # High-resolution image path
            high_res_image_path = os.path.join(args.image_folder, filename)

            # Get the corresponding bicubic and model image paths
            bicubic_image_path = os.path.join(args.image_folder, filename.replace('.tif', f'_bicubic_x{args.scale}.tif'))
            model_image_path = os.path.join(args.image_folder, filename.replace('.tif', f'_{args.model}_x{args.scale}.tif'))

            # Load the images
            high_res_image = cv2.imread(high_res_image_path)
            bicubic_image = cv2.imread(bicubic_image_path)
            model_image = cv2.imread(model_image_path)
            high_res_image = cv2.resize(high_res_image, dsize= (bicubic_image.shape[1],bicubic_image.shape[0]), interpolation=cv2.INTER_CUBIC)


            # Generate masks for each image
            high_res_masks = mask_generator.generate(high_res_image)
            bicubic_masks = mask_generator.generate(bicubic_image)
            model_masks = mask_generator.generate(model_image)

            masks = [high_res_masks, bicubic_masks, model_masks]
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
            non_background_bicubic = all_combined[1]
            non_background_model = all_combined[2]

            edges_gt = canny(non_background_high_res, sigma=1)
            edges_bicubic = canny(non_background_bicubic, sigma=1)
            edges_model = canny(non_background_model, sigma=1)

             # Detect circles in ground truth
            hough_radii = np.arange(50, 180, 2)
            hough_res_gt = hough_circle(edges_gt, hough_radii)
            _, cx_gt, cy_gt, radii_gt = hough_circle_peaks(hough_res_gt, hough_radii, total_num_peaks=4, min_xdistance=250, min_ydistance=300, threshold =0.3 * np.max(hough_res_gt))
    
            # Detect circles in bicubic image
            hough_res_bicubic = hough_circle(edges_bicubic, hough_radii)
            _, cx_bicubic, cy_bicubic, radii_bicubic = hough_circle_peaks(hough_res_bicubic, hough_radii, total_num_peaks=4, min_xdistance=250, min_ydistance=300, threshold = 0.3 * np.max(hough_res_bicubic))
    
            # Detect circles in model image
            hough_res_model = hough_circle(edges_model, hough_radii)
            _, cx_model, cy_model, radii_model = hough_circle_peaks(hough_res_model, hough_radii, total_num_peaks=4, min_xdistance=250, min_ydistance=300, threshold= 0.3 * np.max(hough_res_model))

            

            # Find closest circles based on center points
            tolerance = 10  # You can adjust this tolerance value based on your needs
            matched_indices_gt_b, matched_indices_bicubic = find_closest_circles(cx_gt, cy_gt, cx_bicubic, cy_bicubic, tolerance)
            matched_indices_gt_s, matched_indices_model = find_closest_circles(cx_gt, cy_gt, cx_model, cy_model, tolerance)
            
            # Calculate the difference in radii for matched circles
            differences_bicubic = np.abs(2 * radii_gt[matched_indices_gt_b]* factor - 2 * radii_bicubic[matched_indices_bicubic]* factor)
            differences_model = np.abs(2 * radii_gt[matched_indices_gt_s]* factor - 2 * radii_model[matched_indices_model]* factor)
            
            
            # Add the differences to the respective lists
            relative_differences_bicubic = differences_bicubic / (2 * radii_bicubic[matched_indices_bicubic]* factor)
            relative_differences_model = differences_model / (2 * radii_model[matched_indices_model]* factor)
         
            model_difference.extend(differences_model)
            bicubic_difference.extend(differences_bicubic)
            
            model_difference_relative_list.extend(relative_differences_model)
            bicubic_differences_relative_list.extend(relative_differences_bicubic)
            
            # Calculate number of bubbles detected per each percentage of original
            bicubic_detection = differences_bicubic.size / radii_gt.size
            model_detection = differences_model.size / radii_gt.size
            
            bicubic_detection_list.append(bicubic_detection)
            model_detection_list.append(model_detection)
            
            # Calcualte number of bubbles detected 
            number_bicubic = differences_bicubic.size
            number_model = differences_model.size
            number_gt = radii_gt.size
            
            number_bicubic_list.append(number_bicubic)
            number_model_list.append(number_model)
            number_gt_list.append(number_gt)
            
            
            # Find closest circles based on center points

            radii_gt_zero_filled, radii_bicubic_zero_filled = find_closest_circles_replace_for_0(cx_gt, cy_gt, cx_bicubic, cy_bicubic, radii_gt, radii_bicubic, factor, tolerance)
            radii_gt_zero_filled, radii_model_zero_filled = find_closest_circles_replace_for_0(cx_gt, cy_gt, cx_model, cy_model, radii_gt, radii_model, factor, tolerance)
           
           # Calculate the difference in radii for matched circles
            differences_bicubic_zero = np.abs(radii_gt_zero_filled - radii_bicubic_zero_filled)
            differences_model_zero = np.abs(radii_gt_zero_filled - radii_model_zero_filled)
            
            differences_bicubic_zero_list.extend(differences_bicubic_zero)
            differences_model_zero_list.extend(differences_model_zero)
            

            binary_gt = np.zeros((high_res_image.shape[0], high_res_image.shape[1]), dtype=bool)
            binary_bicubic = np.zeros((bicubic_image.shape[0], bicubic_image.shape[1]), dtype=bool)
            binary_model = np.zeros((model_image.shape[0], model_image.shape[1]), dtype=bool)

            
            for x_gt, y_gt, r_gt in zip(cx_gt, cy_gt, radii_gt):
                yy, xx = np.ogrid[:high_res_image.shape[0], :high_res_image.shape[1]]
                circle_mask = (xx - x_gt) ** 2 + (yy - y_gt) ** 2 <= r_gt ** 2
                binary_gt[circle_mask] = True
            
            for x_b, y_b, r_b in zip(cx_bicubic, cy_bicubic, radii_bicubic):
                yy, xx = np.ogrid[:bicubic_image.shape[0], :bicubic_image.shape[1]]
                circle_mask = (xx - x_b) ** 2 + (yy - y_b) ** 2 <= r_b ** 2
                binary_bicubic[circle_mask] = True
            
            for x_s, y_s, r_s in zip(cx_model, cy_model, radii_model):
                yy, xx = np.ogrid[:model_image.shape[0], :model_image.shape[1]]
                circle_mask = (xx - x_s) ** 2 + (yy - y_s) ** 2 <= r_s ** 2
                binary_model[circle_mask] = True
                
            # Create the masked images
            # boolean indexing and assignment based on mask
            color_img = np.zeros(high_res_image.shape).astype('uint8')
            
            red_mask_high_res = high_res_image.copy()
            red_mask_high_res[binary_gt] = [0, 0, 255]
 
            red_mask_model = model_image.copy()
            red_mask_model[binary_model] = [0,0, 255]   # Set red channel based on mask values
            
            red_mask_bicubic = bicubic_image.copy()
            red_mask_bicubic[binary_bicubic] = [0,0, 255]   # Set red channel based on mask values
            
            # Blend the masked image with the original image
            alpha = 0.4  # Opacity of the red mask (adjust as needed)
            red_mask_high_res = cv2.addWeighted(high_res_image, 1 - alpha, red_mask_high_res, alpha, 0)
            red_mask_model = cv2.addWeighted(model_image, 1 - alpha, red_mask_model, alpha, 0)
            red_mask_bicubic = cv2.addWeighted(bicubic_image, 1 - alpha, red_mask_bicubic, alpha, 0)

            # Save the masked images
            masked_high_res_path = os.path.join(args.image_folder, f"SAM+CHT_{filename}")
            masked_bicubic_path = os.path.join(args.image_folder, f"SAM+CHT_{filename.replace('.tif', f'_bicubic_x{args.scale}.tif')}")
            masked_model_path = os.path.join(args.image_folder, f"SAM+CHT_{filename.replace('.tif', f'_{args.model}_x{args.scale}.tif')}")

            cv2.imwrite(masked_high_res_path, red_mask_high_res)
            cv2.imwrite(masked_bicubic_path, red_mask_bicubic)
            cv2.imwrite(masked_model_path, red_mask_model)



            # Calculate Dice score and IoU
            dice_score_bicubic = calculate_dice_score(binary_gt, binary_bicubic)
            dice_score_model = calculate_dice_score(binary_gt, binary_model)
            iou_bicubic = calculate_iou(binary_gt, binary_bicubic)
            iou_model = calculate_iou(binary_gt, binary_model)

        
            
            # Write the data for each image
            csvwriter_images.writerow([
                filename,
                dice_score_bicubic, dice_score_model,
                iou_bicubic, iou_model,
                2 * radii_gt * factor, 2 * radii_bicubic[matched_indices_bicubic] * factor, 2 * radii_model[matched_indices_model] * factor,
                np.mean(2 * radii_gt * factor), np.mean(2 * radii_bicubic[matched_indices_bicubic] * factor), np.mean(2 * radii_model[matched_indices_model] * factor),
                differences_bicubic, differences_model,
                relative_differences_bicubic, relative_differences_model,
                radii_gt_zero_filled, radii_bicubic_zero_filled, radii_model_zero_filled,
                differences_bicubic_zero, differences_model_zero,
                bicubic_detection, model_detection,
                number_gt, number_bicubic, number_model
            ])

        
            # Append the dice scores and IoU values
            dice_scores_bicubic.append(dice_score_bicubic)
            dice_scores_model.append(dice_score_model)
            ious_bicubic.append(iou_bicubic)
            ious_model.append(iou_model)


    # Calculate overall measures and standard deviations
    overall_dice_bicubic = np.mean(dice_scores_bicubic)
    overall_dice_model = np.mean(dice_scores_model)
    overall_iou_bicubic = np.mean(ious_bicubic)
    overall_iou_model = np.mean(ious_model)

    std_dice_bicubic = np.std(dice_scores_bicubic)
    std_dice_model = np.std(dice_scores_model)
    std_iou_bicubic = np.std(ious_bicubic)
    std_iou_model = np.std(ious_model)

    mean_model_error = np.mean(model_difference)
    std_model_error = np.std(model_difference)
    mean_bicubic_error = np.mean(bicubic_difference)
    std_bicubic_error = np.std(bicubic_difference)
    
    mean_model_relative_error = np.mean(model_difference_relative_list)
    std_model_relative_error = np.std(model_difference_relative_list)
    
    mean_bicubic_relative_error = np.mean(bicubic_differences_relative_list)
    std_bicubic_relative_error = np.std(bicubic_differences_relative_list)
        

    csvwriter_overall.writerow([
        overall_dice_bicubic, std_dice_bicubic,
        overall_dice_model, std_dice_model,
        overall_iou_bicubic, std_iou_bicubic, 
        overall_iou_model, std_iou_model,
        mean_bicubic_error,std_bicubic_error, 
        mean_model_error,std_model_error, 
        mean_bicubic_relative_error, std_bicubic_relative_error, 
        mean_model_relative_error, std_model_relative_error, 
        np.mean(differences_bicubic_zero_list), np.std(differences_bicubic_zero_list),
        np.mean(differences_model_zero_list),np.std(differences_model_zero_list),
        np.mean(bicubic_detection_list), np.std(bicubic_detection_list), 
        np.mean(model_detection_list), np.std(model_detection_list)
      
    ])
    

