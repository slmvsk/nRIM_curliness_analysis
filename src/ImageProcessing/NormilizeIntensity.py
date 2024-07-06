#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 15:52:36 2024

@author: tetianasalamovska
"""

import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt

def normalize_intensity(image_stack):
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

    adjusted_stack = np.zeros_like(image_stack)

    for i in range(image_stack.shape[2]):  # Access each slice in the stack
        img = image_stack[:, :, i]
        adjusted_img = exposure.rescale_intensity(img, in_range='image', out_range=(0, max_val))
        adjusted_stack[:, :, i] = adjusted_img

    return adjusted_stack

def validate_image_adjustment(n_scene, adjusted_stack):
    print("Scene shape:", n_scene.shape)
    scene_min, scene_max = np.min(n_scene), np.max(n_scene)
    print("Scene - min, max:", scene_min, scene_max)

    print("Adjusted stack shape:", adjusted_stack.shape)
    adjusted_min, adjusted_max = np.min(adjusted_stack), np.max(adjusted_stack)
    print("Adjusted stack - min, max:", adjusted_min, adjusted_max)

    if n_scene.dtype == np.uint8:
        expected_max_val = 255
    elif n_scene.dtype == np.uint16:
        expected_max_val = 65535
    else:
        raise ValueError("Unsupported image data type")

    if adjusted_min != 0 or adjusted_max != expected_max_val:
        raise ValueError(f"Adjustment function failed to utilize full dynamic range: Expected 0 to {expected_max_val}, got {adjusted_min} to {adjusted_max}")

    if n_scene.shape != adjusted_stack.shape:
        raise ValueError(f"Shape mismatch: Original shape {n_scene.shape} doesn't match adjusted shape {adjusted_stack.shape}")

def process_all_scenes(scenes):
    for index, scene in enumerate(scenes):
        adjusted_scene = normalize_intensity(scene)
        validate_image_adjustment(scene, adjusted_scene)
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(scene[:, :, scene.shape[2] // 2], cmap='gray')
        plt.title(f'Original Scene {index + 1}')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(adjusted_scene[:, :, adjusted_scene.shape[2] // 2], cmap='gray')
        plt.title(f'Adjusted Scene {index + 1}')
        plt.axis('off')
        plt.show()

# Assuming 'scenes' is your list of 11 3D numpy arrays
#process_all_scenes(scenes)
