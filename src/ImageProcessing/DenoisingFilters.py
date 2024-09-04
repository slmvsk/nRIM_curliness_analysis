#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 10:33:40 2024

@author: tetianasalamovska
"""

import numpy as np
from scipy.ndimage import gaussian_filter
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
#blurred_scenes = apply_gaussian(scenes, sigma=2)
