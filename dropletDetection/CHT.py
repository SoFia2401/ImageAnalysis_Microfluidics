import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import  canny
from skimage.draw import circle_perimeter
import argparse
import matplotlib.font_manager as font_manager
from PIL import Image
import multiprocessing
from functools import partial

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
            
            
    if cx_gt.size ==  0:
        matched_radii_gt = [0]
        matched_radii_pred = [0]
            
            
    return matched_radii_gt, matched_radii_pred


# Function to draw a cross
def draw_cross(image, y, x, length, color, width):
    half_width = width // 2
    image[y - half_width: y + half_width, x - length: x + length] = color
    image[y - length: y + length, x - half_width: x + half_width] = color


# Calculate precision, recall, and F1 score
def calculate_metrics(true_positives, detected, total_gt):
    precision = true_positives / detected if detected > 0 else 0
    recall = true_positives / total_gt if total_gt > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score


# Calculate the distance between two points
def distance(center1, center2):
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)


# Process image for multiprocessing
def process_image(image_folder, tolerance, depthUm, depthPx, bicubic_name, srcnn_name, gt_image_name):
    factor = args.depthUm/args.depthPx
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
    hough_radii = np.arange(50, 150, 2)
    hough_res_gt = hough_circle(edges_gt, hough_radii)
    _, cx_gt, cy_gt, radii_gt = hough_circle_peaks(hough_res_gt, hough_radii, total_num_peaks=4, min_xdistance=250, min_ydistance=300, threshold =0.7 * np.max(hough_res_gt))
    
    # Detect circles in bicubic image
    hough_res_bicubic = hough_circle(edges_bicubic, hough_radii)
    _, cx_bicubic, cy_bicubic, radii_bicubic = hough_circle_peaks(hough_res_bicubic, hough_radii, total_num_peaks=4, min_xdistance=250, min_ydistance=300, threshold = 0.7 * np.max(hough_res_bicubic))

    # Detect circles in SRCNN image
    hough_res_srcnn = hough_circle(edges_srcnn, hough_radii)
    _, cx_srcnn, cy_srcnn, radii_srcnn = hough_circle_peaks(hough_res_srcnn, hough_radii, total_num_peaks=4, min_xdistance=250, min_ydistance=300, threshold= 0.7 * np.max(hough_res_srcnn))


    diameter_hr = np.mean(2*radii_gt)
    
    
    # Find closest circles based on center points
    tolerance = 10  # You can adjust this tolerance value based on your needs
    matched_indices_gt_b, matched_indices_bicubic = find_closest_circles(cx_gt, cy_gt, cx_bicubic, cy_bicubic, tolerance)
    matched_indices_gt_s, matched_indices_model = find_closest_circles(cx_gt, cy_gt, cx_srcnn, cy_srcnn, tolerance)
    
    # Calculate the difference in radii for matched circles
    absolute_diff_bicubic = np.abs(2 * radii_gt[matched_indices_gt_b]* factor - 2 * radii_bicubic[matched_indices_bicubic]* factor)
    absolute_diff_srcnn = np.abs(2 * radii_gt[matched_indices_gt_s]* factor - 2 * radii_srcnn[matched_indices_model]* factor)
    

    relative_diff_bicubic = absolute_diff_bicubic / (2 * radii_bicubic[matched_indices_bicubic]* factor)
    relative_diff_srcnn = absolute_diff_srcnn / (2 * radii_srcnn[matched_indices_model]* factor)
  

    # Calculate number of bubbles detected per each percentage of original
    bicubic_detection = absolute_diff_bicubic.size / radii_gt.size
    model_detection = absolute_diff_srcnn.size / radii_gt.size
    
    
    # Find closest circles based on center points
    radii_gt_zero_filled, radii_bicubic_zero_filled = find_closest_circles_replace_for_0(cx_gt, cy_gt, cx_bicubic, cy_bicubic, radii_gt, radii_bicubic, factor, tolerance)
    radii_gt_zero_filled, radii_model_zero_filled = find_closest_circles_replace_for_0(cx_gt, cy_gt, cx_srcnn, cy_srcnn, radii_gt, radii_srcnn, factor, tolerance)
    

   
   # Calculate the difference in radii for matched circles
    differences_bicubic_zero = np.abs(radii_gt_zero_filled - radii_bicubic_zero_filled)
    differences_model_zero = np.abs(radii_gt_zero_filled - radii_model_zero_filled)
    
    
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
    
    
    # Calculate intersection over union (IoU)
    intersection_over_union_bicubic = np.sum(np.logical_and(binary_bicubic, binary_gt)) / np.sum(np.logical_or(binary_bicubic, binary_gt))
    intersection_over_union_srcnn = np.sum(np.logical_and(binary_srcnn, binary_gt)) / np.sum(np.logical_or(binary_srcnn, binary_gt))

    font_size = 16
    font = plt.matplotlib.font_manager.FontProperties(family='Times New Roman', size=font_size)
    
    images = [gt_image, bicubic_image, srcnn_image]
    titles = ['Original', 'Bicubic', args.model]
    line_width = 3  # Adjust the line width as desired
    
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
    
            for w in range(-line_width, line_width):
                image_with_circles[circy, circx+w] = (34,139,34)
                image_with_circles[circy+w, circx] = (34,139,34)
    
            draw_cross(image_with_circles, center_y, center_x, length=12, color=(34,139,34), width=6)
        
        plt.imshow(image_with_circles)
        plt.axis('off')
    
        # Save the figure as a TIFF image
        output_path = os.path.join(image_folder, f"hough_{gt_image_name}_{titles[i]}.tif")
        plt.savefig(output_path, format='tif', dpi=300)
        plt.close()
    
    
    return gt_image_name, absolute_diff_srcnn, absolute_diff_bicubic, relative_diff_srcnn, relative_diff_bicubic, differences_bicubic_zero, differences_model_zero, bicubic_detection, model_detection, dsc_bicubic, dsc_srcnn, intersection_over_union_bicubic, intersection_over_union_srcnn, diameter_hr


def evaluate_circle_detection(image_folder, scale, depthUm, depthPx, model):
    
    
    tolerance = 20
    bicubic_name = "_bicubic_x" + str(scale) + ".tif"
    srcnn_name = "_" + model + "_x" + str(scale) + ".tif"
    image_files = os.listdir(image_folder)
    gt_images = [file for file in image_files if file.endswith('.tif') and not file.endswith(bicubic_name) and not file.endswith(srcnn_name) and not file.startswith("hough")and not file.startswith("SAM+CHT")]

    dsc_bicubic_list = []
    dsc_srcnn_list = []
    iou_bicubic_list = []
    iou_srcnn_list = []
    srcnn_difference_zero_list =[]
    bicubic_difference_zero_list = []

    srcnn_difference_list = []
    bicubic_difference_list= []
    srcnn_difference_relative_list = []
    bicubic_difference_relative_list = []
    bicubic_detection_list = []
    model_detection_list = []

    
    # Create a log file to save the print statements
    csv_file_images = os.path.join(args.image_folder, "Hough_segmentation_metrics.csv")
    csvwriter_images = open(csv_file_images, 'w', newline='')
    csvwriter_images = csv.writer(csvwriter_images)
    
        # Write the header row
    csvwriter_images.writerow([
        "Image",
        "Dice score - Bicubic", f"Dice score - {args.model}",
        "IoU - Bicubic", f"IoU - {args.model}",
        "Absolute Difference - Bicubic", f"Absolute Difference - {args.model}",
        "Relative Difference - Bicubic", f"Relative Difference - {args.model}",
        "Absolute Difference zero filled - Bicubic", f"Absolute Difference zero filled - {args.model}",
        "Percentage Detection - Bicubic", f"Percentage Detection - {args.model}"
    ])
    
    csv_file_overall = os.path.join(args.image_folder, "Hough_segmentation_metrics_overall.csv")
    csvwriter_overall = open(csv_file_overall, 'w', newline='')
    csvwriter_overall = csv.writer(csvwriter_overall)
    
    # Write the column names for each metric
    csvwriter_overall.writerow([
        "Dice score - Bicubic",
        "Std Dice score - Bicubic",
        f"Dice score - {args.model}",
        f"Std Dice score - {args.model}",
        "IoU - Bicubic",
        "Std IoU - Bicubic",
        f"IoU - {args.model}", 
        f"Std IoU - {args.model}", 
        "Absolute Difference - Bicubic", 
        "Std Absolute Difference - Bicubic", 
        f"Absolute Difference - {args.model}", 
        f"Std Absolute Difference - {args.model}", 
        "Relative Difference - Bicubic", 
        "Std Relative Difference - Bicubic", 
        "Relative Difference - {args.model}", 
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

    


    # For multiprocessing 
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)

    partial_func = partial(process_image, image_folder, tolerance, depthUm, depthPx, bicubic_name, srcnn_name)
    results = pool.map(partial_func, gt_images)

    for result in results:
        gt_image_name, absolute_diff_srcnn, absolute_diff_bicubic, relative_diff_srcnn, relative_diff_bicubic, differences_bicubic_zero, differences_model_zero, bicubic_detection, model_detection, dsc_bicubic, dsc_srcnn, intersection_over_union_bicubic, intersection_over_union_srcnn, diameter_hr = result

        # append results to lists for overall metrics
        dsc_bicubic_list.append(dsc_bicubic)
        dsc_srcnn_list.append(dsc_srcnn)
        iou_bicubic_list.append(intersection_over_union_bicubic)
        iou_srcnn_list.append(intersection_over_union_srcnn)
        srcnn_difference_list.extend(absolute_diff_srcnn)
        bicubic_difference_list.extend(absolute_diff_bicubic)
        srcnn_difference_relative_list.extend(relative_diff_srcnn)
        bicubic_difference_relative_list.extend(relative_diff_bicubic)
        srcnn_difference_zero_list.extend(differences_model_zero)
        bicubic_difference_zero_list.extend(differences_bicubic_zero)
        bicubic_detection_list.append(bicubic_detection)
        model_detection_list.append(model_detection)
        
        
        # Write the results to csv file for individual images
        csvwriter_images.writerow([
            gt_image_name,
            dsc_bicubic, dsc_srcnn,
            intersection_over_union_bicubic, intersection_over_union_srcnn,
            absolute_diff_bicubic, absolute_diff_srcnn,
            relative_diff_bicubic, relative_diff_srcnn,
            differences_bicubic_zero, differences_model_zero,
            bicubic_detection, model_detection,
        ])


    # Calculate overall metrics for the entire test dataset
    overall_dice_bicubic = np.mean(dsc_bicubic_list)
    overall_dice_model = np.mean(dsc_srcnn_list)
    overall_iou_bicubic = np.mean(iou_bicubic_list)
    overall_iou_model = np.mean(iou_srcnn_list)
    
    std_dice_bicubic = np.std(dsc_bicubic_list)
    std_dice_model = np.std(dsc_srcnn_list)
    std_iou_bicubic = np.std(iou_bicubic_list)
    std_iou_model = np.std(iou_srcnn_list)
    
    mean_model_error = np.mean(srcnn_difference_list)
    std_model_error = np.std(srcnn_difference_list)
    mean_bicubic_error = np.mean(bicubic_difference_list)
    std_bicubic_error = np.std(bicubic_difference_list)
    
    mean_model_relative_error = np.mean(srcnn_difference_relative_list)
    std_model_relative_error = np.std(srcnn_difference_relative_list)
    
    mean_bicubic_relative_error = np.mean(bicubic_difference_relative_list)
    std_bicubic_relative_error = np.std(bicubic_difference_relative_list)
    
    
    #write overall metrics to csv
    csvwriter_overall.writerow([
        overall_dice_bicubic, std_dice_bicubic,
        overall_dice_model, std_dice_model,
        overall_iou_bicubic, std_iou_bicubic, 
        overall_iou_model, std_iou_model,
        mean_bicubic_error,std_bicubic_error, 
        mean_model_error,std_model_error, 
        mean_bicubic_relative_error, std_bicubic_relative_error, 
        mean_model_relative_error, std_model_relative_error, 
        np.mean(bicubic_difference_zero_list), np.std(bicubic_difference_zero_list),
        np.mean(srcnn_difference_zero_list),np.std(srcnn_difference_zero_list),
        np.mean(bicubic_detection_list), np.std(bicubic_detection_list), 
        np.mean(model_detection_list), np.std(model_detection_list)
      
    ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-folder', type=str, required=True, help='Path to test images directory')
    parser.add_argument('--scale', type=int, required=True, help='Scale')
    parser.add_argument('--model', type = str, required = True)
    parser.add_argument('--depthUm', type = int, required = True)
    parser.add_argument('--depthPx', type = int, required = True)
    args = parser.parse_args()
    

    evaluate_circle_detection(args.image_folder, args.scale, args.depthUm, args.depthPx, args.model)
