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



import numpy as np
import scipy.ndimage

def calculate_hessian(matrix, sigma):
    """Calculate the Hessian matrix for each voxel in a 3D array."""
    # Calculate the gradients
    gradients = np.gradient(matrix, sigma, axis=(0, 1, 2))
    hessian = np.empty((matrix.shape[0], matrix.shape[1], matrix.shape[2], 3, 3))
    
    # Compute each component of the Hessian matrix
    for k in range(3):
        for l in range(3):
            hessian[..., k, l] = np.gradient(gradients[k], sigma, axis=l)
    
    return hessian

def tubeness(image, sigma):
    """Calculate the tubeness measure of a 3D image using the Hessian matrix eigenvalues."""
    # Gaussian smoothing
    smoothed = scipy.ndimage.gaussian_filter(image, sigma)
    
    # Calculate the Hessian matrix
    hessian = calculate_hessian(smoothed, sigma)
    
    # Compute eigenvalues for each voxel
    eigenvalues = np.linalg.eigvalsh(hessian)
    
    # Select the most negative eigenvalue
    min_eigenvalue = np.min(eigenvalues, axis=-1)
    
    # Tubeness measure: negative eigenvalues indicate tubular structures
    tubeness = np.where(min_eigenvalue < 0, -min_eigenvalue, 0)
    
    return tubeness

# Example usage with a 3D numpy array `data`
# data should be your actual 3D image data
# sigma should be chosen based on the scale of the structures you're looking to enhance
sigma = 4.0  # Gaussian smoothing parameter
result = tubeness(blurred_scenes[2], sigma)

plot_comparison(result[8,:,:], blurred_scenes[2][8,:,:], "Gaussian comparison")


def plot_images(image1, image2, title1='Image 1', title2='Image 2'):
    """Plot two images side by side with high quality."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=300)  # Increase dpi for higher quality
    
    # Plot the first image
    ax[0].imshow(image1, cmap='gray')
    ax[0].set_title(title1)
    ax[0].axis('off')  # Hide the axes

    # Plot the second image
    ax[1].imshow(image2, cmap='gray')
    ax[1].set_title(title2)
    ax[1].axis('off')  # Hide the axes

    plt.tight_layout()  # Adjust layout to fit images
    plt.show()

# Example usage:
# Assuming 'image1' and 'image2' are your 2D numpy arrays representing the images
plot_images(result[8,:,:], blurred_scenes[2][8,:,:], 'Result Slice', 'Blurred Scene Slice')
from scipy.ndimage import gaussian_filter
blurred_result = gaussian_filter(result, sigma=3)

plot_images(blurred_result[8,:,:], blurred_scenes[2][8,:,:], 'Result Slice', 'Blurred Scene Slice')



from skimage.io import imread
from pyclesperanto_prototype import imshow
import pyclesperanto_prototype as cle
import matplotlib.pyplot as plt

import napari
from napari.utils import nbscreenshot

# For 3D processing, powerful graphics
# processing units might be necessary
cle.select_device('TX')
backgrund_subtracted = cle.top_hat_box(blurred_result, radius_x=10, radius_y=10, radius_z=10)
print(blurred_result.shape)

print(backgrund_subtracted.shape)
#not bad but radiuses to be chosen and maybe another method

plot_images(blurred_result[8,:,:], backgrund_subtracted[8,:,:], 'Blurred result Slice', 'Subtrackted background')


# problems with segmentation 
segmented = cle.voronoi_otsu_labeling(backgrund_subtracted, spot_sigma=3, outline_sigma=1)

print(segmented.shape)

plot_images(blurred_result[8,:,:], segmented[8,:,:], 'Blurred result Slice', 'Segmented')


###########################
#If you do not have isotropic pixels or need to perform background corrections
#follow the tutorials from here...
# https://github.com/clEsperanto/pyclesperanto_prototype/blob/master/demo/segmentation/Segmentation_3D.ipynb
###########################

#skeletonize (and project like in fiji or no projection)


from skimage.morphology import skeletonize_3d
from skimage import img_as_ubyte

def skeletonize_image(image):
    """
    Apply skeletonization to a 3D binary image.
    
    Args:
    image (numpy.ndarray): A 3D binary image where the objects are 1's and the background is 0's.
    
    Returns:
    numpy.ndarray: A 3D binary image containing the skeleton of the original image.
    """
    # Ensure the image is in the correct format (binary with values 0 and 1)
    if image.dtype != np.uint8:
        image = img_as_ubyte(image > 0)  # Convert to uint8 and threshold if necessary
    
    # Apply skeletonization
    skeleton = skeletonize_3d(image)
    return skeleton

# Example usage:
# Assume `image_3d` is your 3D numpy array that's already a binary image
skeletonized = skeletonize_image(segmented)

plot_images(blurred_result[8,:,:], skeletonized[8,:,:], 'Blurred result Slice', 'Skeletonized')

print(skeletonized.shape)
save_as_tiff(skeletonized, 'skeletonized.tif')
save_as_tiff(scenes[2], 'scenes_2.tif')


# validation 
# max intensity z - projection

def max_intensity_z_projection(image_3d):
    """
    Create a maximum intensity projection (MIP) of a 3D binary image along the z-axis.
    
    Args:
    image_3d (numpy.ndarray): A 3D numpy array representing the binary image. Expected shape is (z, y, x).
    
    Returns:
    numpy.ndarray: A 2D numpy array representing the maximum intensity projection along the z-axis.
    """
    # Validate that the image is binary (it contains only 0 and 1 values)
    if not np.all(np.isin(image_3d, [0, 1])):
        raise ValueError("Input image must be binary containing only 0 and 1 values.")

    # Compute the maximum intensity projection by using np.max along the z-axis
    mip = np.max(image_3d, axis=0)  # Axis 0 corresponds to the z-direction in your image stack
    return mip



mip_image = max_intensity_z_projection(skeletonized)


print(mip_image.shape)
plot_images(blurred_result[8,:,:], mip_image, 'Blurred result Slice', 'Skeletonized')




#Wrtie image as tif. Ue imageJ for visualization
from skimage.io import imsave
imsave("skeletonized.tif", skeletonized) 

import tifffile as tiff

def save_as_tiff(image_slice, file_name):
    """Save an image slice as a TIFF file."""
    tiff.imwrite(file_name, image_slice, photometric='minisblack')

# Example usage to save specific slices
save_as_tiff(result[8, :, :], 'result_slice_8.tif')
save_as_tiff(blurred_scenes[2][8, :, :], 'blurred_scene_slice_8.tif')





