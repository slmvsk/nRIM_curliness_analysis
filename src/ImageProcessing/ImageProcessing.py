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
    Apply a Gaussian filter to an image.

    Parameters:
        image (ndarray): The input image to filter.
        sigma (float): The sigma (standard deviation) of the Gaussian kernel.

    Returns:
        ndarray: The filtered image.
    """
    # Apply Gaussian filter
    filtered_image = filters.gaussian(image, sigma=sigma)

    return filtered_image

# Apply Gaussian filter with a specific sigma
#filtered_image = apply_gaussian_filter(adjusted_scenes[2], sigma=2)

