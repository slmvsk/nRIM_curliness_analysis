#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 15:52:36 2024

@author: tetianasalamovska
"""

# if converting to 8 bit, do it here 

############### normalize contrast intensities 
# i need not only normalize between images but also for every stack between Z dimension!!!
# then plot averaged intensities on 1 plot colourcoded and compare the output 
# i am normalizing for individual images, manually adjusting "thresholds" that are %les 

import numpy as np
from skimage import exposure, img_as_float


# 2D test
def adjust_image_histogram(image, min_max_thr=(0.1, 0.99)): #0.05, 0.99 gpt sug
    """
    Adjust the histogram of an image based on low and high percentiles.
    
    Parameters:
    image (numpy.ndarray): Input 2D image.
    min_max_thr (tuple): Lower and higher "percentiles" to stretch the intensity range.
    
    Returns:
    numpy.ndarray: Image with adjusted histogram.
    """
    # Convert image to float for percentile calculation
    image_float = img_as_float(image)
    
    # Calculate percentile values
    min_th, max_th = np.percentile(image_float, [min_max_thr[0]*100, min_max_thr[1]*100])
    
    # Adjust the intensity range based on the percentile values
    adjusted_image = exposure.rescale_intensity(image_float, in_range=(min_th), max_th))
    
    return adjusted_image

from skimage import exposure


# 3D (using this one) 
def normalize_intensity_stack(image_stack):
    """
    Normalize intensity of an individual image stack.

    Parameters:
    - image_stack (numpy.ndarray): A 3D numpy array.

    Returns:
    - numpy.ndarray: Adjusted image stack with enhanced contrast.
    """
    if image_stack.dtype == np.uint8:
        max_val = 255
    elif image_stack.dtype == np.uint16:
        max_val = 65535
    else:
        raise ValueError("Unsupported image data type")

    adjusted_stack = np.zeros_like(image_stack, dtype=np.float32)

    # Calculate the percentile values for intensity rescaling
    min_th, max_th = np.percentile(image_stack, [1, 99])  # Use global percentiles for stack or for all images in the list? 

    for i in range(image_stack.shape[0]):  # Access each slice in the stack
        img = image_stack[i, :, :]
        adjusted_img = exposure.rescale_intensity(img, in_range=(min_th, max_th), out_range=(0, max_val))
        adjusted_stack[i, :, :] = adjusted_img

    return adjusted_stack.astype(image_stack.dtype)  # Ensure the output has the same dtype as the input

# Example usage with a hypothetical 3D image stack
#normalized_images = normalize_intensity_stack(test_img)

# Visualize the effect on a middle slice

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(test_img[test_img.shape[0] // 2], cmap='gray')
plt.title('Original Middle Slice')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(normalized_images[normalized_images.shape[0] // 2], cmap='gray')
plt.title('Normalized Middle Slice')
plt.axis('off')
plt.show()



# putting together 
def normalizeScenes(scenes):
    """
    Apply normalization to each 3D numpy array in a list and validate each one.

    Parameters:
    - scenes (list): List of 3D numpy arrays.

    Returns:
    - list: List of adjusted 3D numpy arrays.
    """
    adjusted_scenes = []
    for scene in scenes:
        adjusted_scene = normalize_intensity_stack(scene) #fixed
        validateImageAdjustment(scene, adjusted_scene)
        adjusted_scenes.append(adjusted_scene)
    return adjusted_scenes

def validateImageAdjustment(scene, adjusted_scene):
    print("Scene shape:", scene.shape)
    scene_min, scene_max = np.min(scene), np.max(scene)
    print("Scene - min, max:", scene_min, scene_max)

    print("Adjusted scene shape:", adjusted_scene.shape)
    adjusted_min, adjusted_max = np.min(adjusted_scene), np.max(adjusted_scene)
    print("Adjusted scene - min, max:", adjusted_min, adjusted_max)

    if scene.dtype == np.uint8:
        expected_max_val = 255
    elif scene.dtype == np.uint16:
        expected_max_val = 65535
    else:
        raise ValueError("Unsupported image data type")

    if adjusted_min != 0 or adjusted_max != expected_max_val:
        raise ValueError(f"Adjustment function failed to utilize full dynamic range: Expected 0 to {expected_max_val}, got {adjusted_min} to {adjusted_max}")

    if scene.shape != adjusted_scene.shape:
        raise ValueError(f"Shape mismatch: Original shape {scene.shape} doesn't match adjusted shape {adjusted_scene.shape}")

# Assuming 'scenes' is your list of 11 3D numpy arrays
#normalized_scenes = normalizeScenes(scenes)


#plot_images(normalized_scenes[3][8,:,:], scenes[3][8,:,:], 'nm', 'orig')


















