#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 14:04:44 2024

@author: tetianasalamovska
"""

import numpy as np
from skimage.morphology import white_tophat, ball
from skimage import morphology

def subtractBackground(image_stack, radius=10):
    """
    Subtract background using a white top-hat filter on a 3D image stack.

    Parameters:
        image_stack (numpy.ndarray): A 3D numpy array of the image stack.
        radius (int): The radius of the structuring element to use in the top-hat filter.

    Returns:
        numpy.ndarray: 3D image stack with background subtracted.
    """
    # Create a 2D disk structuring element for background subtraction
    footprint = morphology.disk(radius)

    # Initialize the output array
    background_subtracted_stack = np.empty_like(image_stack)

    # Apply white top-hat filtering to each slice in the stack
    for i in range(image_stack.shape[0]):
        background_subtracted_stack[i, :, :] = white_tophat(image_stack[i, :, :], footprint=footprint)

    return background_subtracted_stack


def subtractBackgroundFromScenes(scenes, radius=5):
    """
    Apply background subtraction to each 3D numpy array in a list using the white top-hat filter.
    This function aims to be memory-efficient by operating in place and cleaning up after each operation.

    Parameters:
        scenes (list): List of 3D numpy arrays where each array represents a scene.
        radius (int): The radius of the structuring element to use in the top-hat filter.

    Returns:
        list: A list of 3D numpy arrays with the background subtracted.
    """
    processed_scenes = []
    for index, scene in enumerate(scenes):
        print(f"Processing scene {index + 1}/{len(scenes)}...")
        
        # Apply the background subtraction directly to the scene
        background_subtracted_scene = subtractBackground(scene.copy(), radius=radius)  # copy only if necessary
        
        # Store the processed scene
        processed_scenes.append(background_subtracted_scene)
        
        # Explicitly delete the temporary variables to free up memory
        del background_subtracted_scene
        print(f"Scene {index + 1} processed.")
        
    return processed_scenes

# Assuming 'scenes' is your list of 3D numpy arrays
#subtracted_scenes = subtractBackgroundFromScenes(scenes, radius=25)
