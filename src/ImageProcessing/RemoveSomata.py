#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 00:40:39 2024

@author: tetianasalamovska
"""

# thresholding = denoising! Otsu?  ####### make 8 bit before this step 
##############################################

# Operations like enhancing contrast or thresholding can be done as for 2D 
# images where the context of adjacent slices isn’t important ? 
# Segmentation must be done for scenes not for slices 

import numpy as np
from skimage.filters import threshold_multiotsu
from skimage import img_as_float


# use median filtering here 
from skimage.filters import median
from skimage.morphology import ball
import numpy as np

def apply_median_filter_to_scenes(scenes, radius=1):
    """
    Apply a 3D median filter to each scene in a list of 3D numpy arrays.

    Parameters:
        scenes (list of numpy.ndarray): List of 3D numpy arrays representing different scenes.
        radius (int): Radius of the ball-shaped structuring element used for the median filter.

    Returns:
        list of numpy.ndarray: List of filtered 3D numpy arrays.
    """
    filtered_scenes = []
    struct_element = ball(radius)  # Create a ball structuring element for 3D median filtering

    for scene in scenes:
        filtered_scene = median(scene, struct_element)
        filtered_scenes.append(filtered_scene)

    return filtered_scenes

# Example usage:
# Assuming 'scenes' is your list of 3D numpy arrays
#radius = 2  # You can adjust the radius based on your specific noise characteristics and image resolution
#filtered_scenes = apply_median_filter_to_scenes(scenes, radius)





def findOptimalThreshold(img, metric_th=0.9):
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



from skimage import img_as_float
from skimage.filters import threshold_multiotsu
from skimage.filters import median
from skimage.morphology import ball  # Use ball for 3D structuring element
import numpy as np

def removeSomafromStack(image_stack, xy_resolution):
    """Remove somas from an image stack based on intensity thresholds."""
    img_float = img_as_float(image_stack)  # Ensure the image is in floating point
    n_slices = image_stack.shape[2]
    
    # Apply a 3D median filter to the entire stack
    # Use a small ball as the structuring element; you might adjust the size based on your specific needs
    img_filtered = median(img_float, ball(1))  # ball(1) provides a reasonable balance in 3D

    # Determine the optimal threshold on the filtered middle slice
    th_lvl = findOptimalThreshold(img_filtered[:, :, n_slices // 2])

    # Apply multi-level thresholding
    thresholds = threshold_multiotsu(img_filtered[:, :, n_slices // 2], classes=th_lvl)
    quant_a = np.digitize(img_filtered, bins=thresholds)
    
    # Create background mask
    bg_mask = quant_a <= th_lvl * 0.2  # More aggressive background masking
    
    # Filter image stack: set background regions to zero
    image_stack_filtered = np.copy(image_stack)  # Copy the original stack to maintain original data type
    for i in range(n_slices):
        image_stack_filtered[:, :, i][bg_mask[:, :, i]] = 0

    return image_stack_filtered


nosoma_img = removeSomafromStack(normalized_scenes[5], xy_resolution=1)
img_filtered = median(normalized_scenes[5], ball(3))  # ball(1) provides a reasonable balance in 3D
nosoma_img_med = removeSomafromStack(img_filtered, xy_resolution=1)
plot_images(img_filtered[8,:,:], nosoma_img_med[8,:,:], 'Original', 'No soma')



#debugging step 
def removeSomaFromAllScenes(scenes, xy_resolution):
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
            processed_scene = removeSomafromStack(scene, xy_resolution)
            processed_scenes.append(processed_scene)
            print(f"Processed scene {i+1} successfully added to the list.")
              
        except Exception as e:
            print(f"Error processing scene {i+1}: {e}")
            continue
        # Release memory for the current scene
        del scene
    print(f"Total processed scenes: {len(processed_scenes)}")
    return processed_scenes

nosoma_scenes = removeSomaFromAllScenes(normalized_scenes, xy_resolution=1)
print(f"Number of scenes processed and returned: {len(nosoma_scenes)}")


# Optionally, inspect the first scene to ensure it's not empty
if len(nosoma_scenes) > 0:
    print(f"Shape of the first processed scene: {nosoma_scenes[0].shape}")
else:
    print("No scenes were processed.")





from skimage.filters import median, gaussian
from skimage.morphology import disk



####################################
# yes it is for 0ne scene only, but all my functions will be for 1 scene and then 
# I will just iterate over all of the scenes and files?????????????????
#or make a function that will iterates for files but not scenes, that will 
# be already made for scenes 
# it is also a question that i will release memory after each scene or after each file?
####################################


# Example usage
# Assume `image_stack` is your 3D numpy array with shape [height, width, depth]
# `xy_resolution` is a parameter that you might use to adjust algorithm behavior based on image resolution


plot_images(normalized_scenes[5][8,:,:], nosoma_img[8,:,:], 'Original', 'No soma')



# geometrical approach or deep learningg approach to remove leftovers 

# for my example i need boinarise manually
# try Z-projection 
#mip_image = max_intensity_z_projection(image_nosoma)





