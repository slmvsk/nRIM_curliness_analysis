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

def morphological_skeleton_3d(image):
    return skimage.morphology.skeletonize_3d(image)


from ...imageprocessing import morphological_skeleton_2d, morphological_skeleton_3d

def morphologicalskeleton(image, volumetric):
    if volumetric: 
        return morphological_skeleton_3d(image)
    else:
        return morphological_skeleton_2d(image)


def median_filter(image, window_size, mode):
    return scipy.ndimage.median_filter(image, size=window_size, mode=mode)


def reduce_noise(image, patch_size, patch_distance, cutoff_distance, channel_axis=None):
    denoised = skimage.restoration.denoise_nl_means(
        image=image,
        patch_size=patch_size,
        patch_distance=patch_distance,
        h=cutoff_distance,
        channel_axis=channel_axis,
        fast_mode=True,
    )
    return denoised



# 2D to try because takes time for 3D 
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

import numpy as np
from skimage import filters, io
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

def apply_tubeness_3d(image, sigma_values):
    """
    Compute the tubeness measure for a 3D image using the definition from Sato et al., 1997.

    Parameters:
        image (numpy.ndarray): The input 3D image.
        sigma_values (list): List of sigma values for scale-space analysis.

    Returns:
        numpy.ndarray: A 3D array containing the tubeness measure.
    """
    # Initialize the tubeness image with zeros
    tubeness_image = np.zeros_like(image, dtype=np.float64)

    # Iterate over the range of sigma values
    for sigma in sigma_values:
        # Compute the Hessian matrix
        hessian = hessian_matrix(image, sigma=sigma, order='rc')
        # Compute the eigenvalues of the Hessian matrix
        hessian_eigenvalues = hessian_matrix_eigvals(hessian)
        # Sort eigenvalues by magnitude (largest in magnitude last so lambda2 is -2 and lambda3 is -1 index)
        sorted_eigenvalues = np.sort(np.abs(hessian_eigenvalues), axis=0)

        # Calculate tubeness measure
        is_tubular = (sorted_eigenvalues[-2] < 0) & (sorted_eigenvalues[-1] < 0)
        tubeness_measure = np.sqrt(sorted_eigenvalues[-2] * sorted_eigenvalues[-1])
        tubeness_measure[~is_tubular] = 0

        # Update the tubeness image with the maximum response across scales
        tubeness_image = np.maximum(tubeness_image, tubeness_measure)

    return tubeness_image

sigma_values = [5, 6, 7, 8, 9]
# Example usage
enhanced_stack = apply_tubeness_3d(blurred_scenes[2], sigma_values)  # Example sigma values for scale-space analysis)
print(blurred_scenes[2].shape)
print(enhanced_stack.shape)

plot_comparison(enhanced_stack[8,:,:], blurred_scenes[2][8,:,:], "Gaussian comparison")



def apply_tubeness_3d_debug(image, sigma_values, threshold=0.5):
    tubeness_image = np.zeros_like(image, dtype=np.float64)
    for sigma in sigma_values:
        hessian = hessian_matrix(image, sigma=sigma, order='rc')
        eigenvalues = hessian_matrix_eigvals(hessian)
        sorted_eigenvalues = np.sort(np.abs(eigenvalues), axis=0)
        
        is_tubular = (sorted_eigenvalues[-2] < 0) & (sorted_eigenvalues[-1] < 0)
        tubeness_measure = np.sqrt(sorted_eigenvalues[-2] * sorted_eigenvalues[-1])
        tubeness_measure[~is_tubular] = 0
        
        tubeness_image = np.maximum(tubeness_image, tubeness_measure)
        
        # Debug output
        print("Max Tubeness at sigma =", sigma, ":", tubeness_measure.max())
    
    # Applying a global threshold might be revisited
    tubeness_image[tubeness_image < threshold] = 0
    return tubeness_image

# Usage example with debug
sigma_values = [5, 8, 10, 12, 15]
enhanced_stack = apply_tubeness_3d_debug(blurred_scenes[2], sigma_values)

plot_comparison(enhanced_stack[8,:,:], blurred_scenes[2][8,:,:], "Gaussian comparison")






def enhance_edges_log(image, mask=None, sigma=2.0):
    size = int(sigma * 4) + 1
    output_pixels = centrosome.filter.laplacian_of_gaussian(image, mask, size, sigma)
    return output_pixels


