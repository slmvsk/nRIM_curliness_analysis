#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 07:27:03 2024

@author: tetianasalamovska
"""

from scipy.ndimage import gaussian_filter

from skimage import filters

def apply_gaussian(image, sigma=1.0):
    """
    Applies a Gaussian filter to every slice in every scene.

    Parameters:
        scenes (list of ndarray): List of 3D numpy arrays where each array represents a scene.
        sigma (float): The sigma (standard deviation) of the Gaussian kernel.

    Returns:
        list of ndarray: A list of 3D numpy arrays with the Gaussian filter applied to each slice.
    """
    filtered_scenes = []
    for scene in scenes:
        # Apply Gaussian filter to each slice in the scene
        filtered_scene = np.empty_like(scene)
        for i in range(scene.shape[2]):
            filtered_scene[:, :, i] = filters.gaussian(scene[:, :, i], sigma=sigma)
        filtered_scenes.append(filtered_scene)
    return filtered_scenes

# Apply Gaussian filter with a specific sigma
#filtered_image = apply_gaussian_filter(adjusted_scenes[2], sigma=2)

