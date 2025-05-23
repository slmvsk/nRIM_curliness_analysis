#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 00:40:39 2024

@author: tetianasalamovska
"""

import numpy as np
from skimage.filters import threshold_multiotsu
from skimage import img_as_float
#import matplotlib.pyplot as plt
#from skimage import data
#from skimage import color, morphology
#from skimage import restoration, exposure
from scipy.ndimage import label
import scipy.ndimage as ndimage


# use this function below (manual one) for now for good segmentation 

# for 1 stack (3D)
def binarizeImage(image_stack, threshold):
    """
    Binarize a 3D image stack based on a given threshold value.
    
    Parameters:
        image_stack (ndarray): A 3D numpy array representing the image stack.
        threshold (float): Threshold value for binarization.
    
    Returns:
        ndarray: A 3D numpy array of the binarized image stack.
    """
    # Initialize a binarized stack
    binarized_stack = np.zeros_like(image_stack, dtype=bool)
    
    # Binarize each slice
    for i in range(image_stack.shape[0]):
        img_float = img_as_float(image_stack[i, :, :])  # Convert slice to float
        binarized_stack[i, :, :] = img_float > threshold  # Apply threshold
    
    return binarized_stack


# for all scenes 
def removeSomaFromAllScenes(scenes, thresholds):
    """
    Iterate over all scenes in a file, apply the removeSomafromStack function to each scene,
    and release memory after processing each scene.
    
    Parameters:
        scenes (list): List of 3D numpy arrays where each array represents a scene.
        xy_resolution (float): Resolution scaling factor in the XY plane.
    
    Returns:
        list: A list of 3D numpy arrays with somas removed.
    """
    processed_scenes = []
    
    for i, scene in enumerate(scenes):
        print(f"Processing scene {i+1}/{len(scenes)}")
        
        # Check if scene is valid
        if scene.size == 0:
            print(f"Scene {i+1} is empty or invalid!")
            continue
        
        try:
            # Apply the removeSomafromStack function to the current scene
            processed_scene = binarizeImage(scene, thresholds) #changed to manual temporary 
            processed_scenes.append(processed_scene)
            print(f"Processed scene {i+1} successfully added to the list.")
              
        except Exception as e:
            print(f"Error processing scene {i+1}: {e}")
            continue
        # Release memory for the current scene
        del scene
    print(f"Total processed scenes: {len(processed_scenes)}")
    return processed_scenes





# code USED BEFORE FOR AUTOMATIC THRESHOLDING (finding optimal levels) 
def findOptimalThreshold(img, metric_th=0.95):
    """Determine the optimal number of threshold levels based on a target metric threshold."""
    metrics = []
    optimal_th = 1
    for th_lvl in range(1, 11):  # Test from 1 to 10 levels
        thresholds = threshold_multiotsu(img, classes=th_lvl)
        # Calculate a metric for these thresholds; here we use a simple placeholder
        # In practice, you'd want a metric that evaluates segmentation quality
        metric = np.var(thresholds) / np.mean(thresholds)
        metrics.append(metric)
        if metric > metric_th:
            optimal_th = th_lvl
            break
    else:
        # If no threshold level meets the threshold metric, pick the one with the highest metric
        optimal_th = np.argmax(metrics) + 1
    return optimal_th
def removeSomafromStack(image_stack, xy_resolution):
    """Remove somas from an image stack based on intensity thresholds."""
    img_float = img_as_float(image_stack)  # Ensure the image is in floating point
    n_slices = image_stack.shape[2]
    th_lvl = findOptimalThreshold(image_stack[:, :, n_slices // 2])
    
    # Apply multi-level thresholding
    thresholds = threshold_multiotsu(img_float[:, :, n_slices // 2], classes=th_lvl)
    quant_a = np.digitize(img_float, bins=thresholds)
    
    # Create background mask
    bg_mask = quant_a <= th_lvl * 0.2 # * 0.3 is fine 
    
    # Filter image stack: set background regions to zero
    image_stack_filtered = np.copy(image_stack)
    for i in range(n_slices):
        image_stack_filtered[:, :, i][bg_mask[:, :, i]] = 0

    return image_stack_filtered









### NEXT STEP IS TO REMOVE SMALL OBJECTS ################################

from scipy.ndimage import label
import scipy.ndimage as ndimage

def remove_small_objects_3d(binary_image, min_size=50):
    """
    Remove small objects from a 3D binary image (values 0 and 1) based on their size.

    Parameters:
        binary_image (numpy.ndarray): A 3D binary numpy array with values 0 and 1.
        min_size (int): The minimum size of objects to keep.

    Returns:
        numpy.ndarray: A 3D binary image with small objects removed.
    """
    # Label the binary image with connectivity defining how pixels are connected
    labeled_image, num_features = label(binary_image, structure=ndimage.generate_binary_structure(3, 2))

    # Remove small objects
    unique, counts = np.unique(labeled_image, return_counts=True)
    remove = unique[counts < min_size]
    for obj in remove:
        labeled_image[labeled_image == obj] = 0

    # Create a cleaned binary image
    cleaned = labeled_image > 0
    
    return cleaned


def cleanBinaryScenes(scenes, min_size=50):
    """
    Apply small object removal to each 3D image stack in the scenes.

    Parameters:
        scenes (list of numpy.ndarray): A list of 3D numpy arrays representing the scenes (stacks).
        min_size (int): The minimum size of objects to keep in the binary image.

    Returns:
        list of numpy.ndarray: A list of processed 3D binary numpy arrays.
    """
    processed_scenes = []

    for i, scene in enumerate(scenes):
        print(f"Processing scene {i+1}/{len(scenes)}")

        # Ensure that the scene is binary
        binary_image = np.where(scene > 0, 1, 0)  # Convert to binary (if not already binary)

        # Remove small objects
        final_binary_image = remove_small_objects_3d(binary_image, min_size=min_size)

        # Append processed image to the list
        processed_scenes.append(final_binary_image)
    
    return processed_scenes



