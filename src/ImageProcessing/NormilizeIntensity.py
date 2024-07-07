#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 15:52:36 2024

@author: tetianasalamovska
"""

#from skimage import exposure

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

def process_scenes(scenes):
    """
    Apply normalization to each 3D numpy array in a list and validate each one.

    Parameters:
    - scenes (list): List of 3D numpy arrays.

    Returns:
    - list: List of adjusted 3D numpy arrays.
    """
    adjusted_scenes = []
    for scene in scenes:
        adjusted_scene = normalize_intensity(scene)
        validate_image_adjustment(scene, adjusted_scene)
        adjusted_scenes.append(adjusted_scene)
    return adjusted_scenes

def validate_image_adjustment(scene, adjusted_scene):
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
#adjusted_scenes = process_scenes(scenes)
