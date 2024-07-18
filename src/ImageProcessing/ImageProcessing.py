#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 07:27:03 2024

@author: tetianasalamovska
"""


# Apply Gaussian filter with a specific sigma
#filtered_image = apply_gaussian_filter(adjusted_scenes[2], sigma=2)

import numpy as np
from skimage.filters import gaussian, threshold_otsu
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage import exposure, morphology

def enhance_neurites(image, sigma):
    """
    Enhance neurites using an advanced method incorporating Gaussian smoothing, 
    Hessian-based tubeness, contrast enhancement, and thresholding.

    Parameters:
    image (np.ndarray): The input image (2D).
    sigma (float): The standard deviation for the Gaussian kernel.

    Returns:
    np.ndarray: The processed image highlighting neurites.
    """
    # Smooth the image with a Gaussian filter
    smoothed_image = gaussian(image, sigma=sigma)

    # Compute the Hessian matrix and its eigenvalues
    hessian = hessian_matrix(smoothed_image, sigma=sigma)
    eigenvalues = hessian_matrix_eigvals(hessian)

    # Find the maximum negative eigenvalue
    filtered_image = np.max(eigenvalues, axis=0)
    filtered_image[filtered_image > 0] = 0
    filtered_image = np.abs(filtered_image)

    # Enhance contrast
    contrast_enhanced_image = exposure.equalize_adapthist(filtered_image)

    # Thresholding
    threshold_value = threshold_otsu(contrast_enhanced_image)
    binary_image = contrast_enhanced_image > threshold_value

    # Morphological closing to remove small holes and use dilation to enhance visibility
    final_image = morphology.dilation(morphology.closing(binary_image, morphology.disk(3)), morphology.disk(1))

    return final_image

# Example usage, assuming 'img' is your loaded 2D image array
# img_enhanced = enhance_neurites(img, sigma=2)