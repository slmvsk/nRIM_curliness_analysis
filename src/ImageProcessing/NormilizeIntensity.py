#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 09:13:34 2024

@author: tetianasalamovska
"""

import numpy as np
import cv2
from skimage import exposure
import matplotlib.pyplot as plt

# make it for all scenes 

def normalize_intensity(n_scene):
    """
    Normalize intensity of an individual image stack.

    Parameters:
    - image_stack (numpy.ndarray): A 3D numpy array.

    Returns:
    - numpy.ndarray: Adjusted image stack with enhanced contrast.
    """
    # Determine the data type for setting max intensity value
    if n_scene.dtype == np.uint8:
        max_val = 255
    elif n_scene.dtype == np.uint16:
        max_val = 65535
    else:
        raise ValueError("Unsupported image data type")

    adjusted_stack = np.zeros_like(n_scene)

    # Process each image in the stack
    for i in range(n_scene.shape[2]): # in python you count from 0, 2 is depth(N of slices)
        # Retrieve single image slice
        img = n_scene[:, :, i]
        
        # Use skimage's exposure module to rescale the intensities
        adjusted_img = exposure.rescale_intensity(img, in_range='image', out_range=(0, max_val))
        
        # Store adjusted image back in the stack
        adjusted_stack[:, :, i] = adjusted_img

    return adjusted_stack

# Example of using the function

# Adjust the histogram of the first scene
#adjusted_stack = adjust_histogram(first_scene)

# Display the original and adjusted images of the first scene for comparison



def validate_image_adjustment(n_scene, adjusted_stack):
    print("N scene shape:", n_scene.shape)
    n_scene_min, n_scene_max = np.min(n_scene[:, :, 0]), np.max(n_scene[:, :, 0])
    print("N slice of the n scene - min, max:", n_scene_min, n_scene_max)

    print("Adjusted stack shape:", adjusted_stack.shape)
    adjusted_min, adjusted_max = np.min(adjusted_stack[:, :, 0]), np.max(adjusted_stack[:, :, 0])
    print("N slice of adjusted stack - min, max:", adjusted_min, adjusted_max)

    # Determine the expected max value based on data type
    if n_scene.dtype == np.uint8:
        expected_max_val = 255
    elif n_scene.dtype == np.uint16:
        expected_max_val = 65535
    else:
        raise ValueError("Unsupported image data type")

    # Check if the adjusted stack uses the full dynamic range
    if adjusted_min != 0 or adjusted_max != expected_max_val:
        raise ValueError("Adjustment function failed to utilize full dynamic range: Expected 0 to {}, got {} to {}".format(
            expected_max_val, adjusted_min, adjusted_max_val))

    # Check if the shapes are consistent
    if n_scene.shape != adjusted_stack.shape:
        raise ValueError("Shape mismatch: Original shape {} doesn't match adjusted shape {}".format(
            n_scene.shape, adjusted_stack.shape))

# Example usage of the function
# Assuming 'first_scene' and 'adjusted_stack' are already defined
# validate_image_adjustment(first_scene, adjusted_stack)
