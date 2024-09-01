#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 15:52:36 2024

@author: tetianasalamovska
"""

from skimage.measure import label, regionprops
from skimage import morphology, img_as_float, exposure


def linearContrastStretching(image_stack, percentiles=[0.1, 99.9]):
    """
    Apply linear contrast stretching to an entire 3D image stack globally with adjustable percentiles.

    Parameters:
    - image_stack (numpy.ndarray): A 3D numpy array (Z, Y, X) of an image stack.
    - percentiles (list): List of two values (lower, upper) to define the stretch limits.

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
    min_th, max_th = np.percentile(image_stack_float, percentiles)

    # Apply contrast stretching across the stack
    adjusted_stack = np.zeros_like(image_stack_float)
    for i in range(image_stack.shape[0]):  # Adjust each slice
        adjusted_stack[i, :, :] = exposure.rescale_intensity(
            image_stack_float[i, :, :], in_range=(min_th, max_th), out_range=output_range
        )

    return adjusted_stack.astype(image_stack.dtype)  # Convert back to original data type



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



# putting above 2 function together to process all scenes in the file
def normalizeScenes(scenes, perceltiles=[0.1, 99.9]):
    """
    Apply normalization to each 3D numpy array in a list and validate each one.

    Parameters:
    - scenes (list): List of 3D numpy arrays.

    Returns:
    - list: List of adjusted 3D numpy arrays.
    """
    adjusted_scenes = []
    for scene in scenes:
        adjusted_scene = linearContrastStretching(scene, percentiles) 
        validateImageAdjustment(scene, adjusted_scene)
        adjusted_scenes.append(adjusted_scene)
    return adjusted_scenes


# Assuming 'scenes' is your list of 11 3D numpy arrays
# normalized_scenes = normalizeScenes(scenes)

