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


# 3D (using this one) 

import numpy as np
from skimage import exposure
from skimage.measure import label, regionprops
from skimage import morphology, img_as_float

def linear_contrast_stretching(image_stack):
    """
    Apply linear contrast stretching to an entire 3D image stack globally.

    Parameters:
    - image_stack (numpy.ndarray): A 3D numpy array (Z, Y, X) of an image stack.

    Returns:
    - numpy.ndarray: Contrast stretched 3D image stack.
    """
    # Determine the data type and set the output range accordingly
    if image_stack.dtype == np.uint8:
        output_range = (0, 255)
    elif image_stack.dtype == np.uint16:
        output_range = (0, 65535)
    else:
        raise ValueError("Unsupported image data type")


    # Convert the image stack to floating point to handle intensity scaling
    image_stack_float = img_as_float(image_stack)

    # Calculate global percentiles across the entire stack to handle outliers
    # Exclude the top and bottom 1% of data to reduce the impact of outliers
    min_th, max_th = np.percentile(image_stack_float, [0.1, 99.9])

    # Apply contrast stretching across the stack
    adjusted_stack = np.zeros_like(image_stack_float)
    for i in range(image_stack.shape[0]):  # Adjust each slice
        adjusted_stack[i, :, :] = exposure.rescale_intensity(
            image_stack_float[i, :, :], in_range=(min_th, max_th), out_range=output_range
        )

    return adjusted_stack.astype(image_stack.dtype)  # Convert back to original data type

# diagnostic 
print("Min pixel value:", np.min(scenes[4]))
print("Max pixel value:", np.max(scenes[4]))


# Contrast stretching






# Optionally visualize or further process `enhanced_stack`
plot_image_histogram(img_adapteq[8,:,:])
plot_images(normalized_scenes[4][8,:,:], enhanced_stack[8,:,:], 'Original', 'No soma')


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
        adjusted_scene = linear_contrast_stretching(scene) #fixed
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
normalized_scenes = normalizeScenes(scenes)





plot_images(normalized_scenes[9][18,:,:], scenes[9][18,:,:], 'nm', 'orig')

# equalization do not use, i need to preserve intensities in the images to threshold

import numpy as np
import matplotlib.pyplot as plt

def plot_image_histogram(image, bins=256, title='Image Histogram', max_intensity=70000):
    """
    Plot the histogram of an image.

    Parameters:
        image (numpy.ndarray): A 2D numpy array representing the grayscale image.
        bins (int): Number of histogram bins.
        title (str): Title of the histogram plot.
        max_intensity (int): Maximum intensity to include in the histogram.
    """
    # Calculate histogram
    histogram, bin_edges = np.histogram(image, bins=bins, range=[0, max_intensity])

    # Configure plot
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Pixel Count')

    # Plot the histogram
    plt.bar(bin_edges[:-1], histogram, width=np.diff(bin_edges), edgecolor='black', align='edge')

    # Display the plot
    plt.show()

# Example usage with your image data:


# Example usage
plot_image_histogram(skeletonized)




