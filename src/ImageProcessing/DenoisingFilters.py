#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 10:33:40 2024

@author: tetianasalamovska
"""

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from skimage.filters import gaussian


def applyGaussian(scenes, sigma=1):
    """
    Apply Gaussian blur to all scenes in a list of 3D image stacks.
    
    Parameters:
        scenes (list of numpy.ndarray): List of 3D numpy arrays where each array represents a scene.
        sigma (float or sequence of floats): Standard deviation for Gaussian kernel.
    
    Returns:
        list of numpy.ndarray: List of 3D numpy arrays with Gaussian blur applied.
    """
    blurred_scenes = []
    for i, scene in enumerate(scenes):
        # Apply Gaussian blur to the current scene
        blurred_scene = gaussian_filter(scene, sigma=sigma)
        blurred_scenes.append(blurred_scene)
        print(f"Applied Gaussian blur to scene {i+1}/{len(scenes)}")
    
    return blurred_scenes

# Example usage:
# Assuming 'scenes' is your list of 3D numpy arrays
#blurred_scenes = applyGaussian(scenes, sigma=2)


def applyMedianFilter(scenes, size=3):
    """
    Apply a median filter to all scenes in a list of 3D image stacks.
    
    Parameters:
        scenes (list of numpy.ndarray): List of 3D numpy arrays where each array represents a scene.
        size (int or sequence of ints): Size of the median filter. The filter size should be odd and positive.
    
    Returns:
        list of numpy.ndarray: List of 3D numpy arrays with the median filter applied.
    """
    filtered_scenes = []
    for i, scene in enumerate(scenes):
        # Apply median filter to the current scene
        filtered_scene = median_filter(scene, size=size)
        filtered_scenes.append(filtered_scene)
        print(f"Applied median filter to scene {i+1}/{len(scenes)}")
    
    return filtered_scenes

# Example usage:
#median_scenes = applyMedianFilter(scenes, size=3)



