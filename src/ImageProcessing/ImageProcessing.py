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
import scipy.ndimage
from skimage.io import imread
from pyclesperanto_prototype import imshow
import matplotlib.pyplot as plt
import napari
from napari.utils import nbscreenshot
import pyclesperanto_prototype as cle


# max z projection of nosoma_image + size-based filter function 

#
#
#






##########################
# Tubeness problem:
# 1. It wants to see image with only tubular structures, so sometimes it connects dots
# 2. Also sigma is dependent on the scale of image so I need to make function "smart" to be
# able to accept different scale as input and adjust sigma?
# Solution: 
# 1. Preprocess better, dots and small onjects doesn't give me any information (dendrite pieces)
# 2. 
##########################


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
    hessian = calculate_hessian(smoothed, sigma) #replace with smoothed if previous step is active 
    
    # Compute eigenvalues for each voxel
    eigenvalues = np.linalg.eigvalsh(hessian)
    
    # Select the most negative eigenvalue
    min_eigenvalue = np.min(eigenvalues, axis=-1)
    
    # Tubeness measure: negative eigenvalues indicate tubular structures
    tubeness = np.where(min_eigenvalue < 0, -min_eigenvalue, 0)
    
    return tubeness

#validation of tubness 
#normalize intensity and substract tubeness image from original
# ......


def estimate_sigma_based_on_scale(image, scale_factor=1.0):
    """
    Estimate an appropriate sigma for the tubeness function based on the image scale.
    
    Parameters:
        image (ndarray): The input 3D image array.
        scale_factor (float): A multiplier for scaling sigma according to the image resolution.
    
    Returns:
        float: The estimated sigma value.
    """
    # Example estimation based on image size and scale factor
    median_dim = np.median(image.shape)
    sigma = median_dim * 0.005 * scale_factor    
    return sigma

# Example usage:
  # Replace this with your actual 3D image
sigma = estimate_sigma_based_on_scale(nosoma_scenes[5], scale_factor=0.9)
tubeness_measure = tubeness(nosoma_scenes[5], sigma)


plot_images(result[8,:,:], nosoma_scenes[8][8,:,:], 'Original', 'No Somata')

# more or less but 1. needs validation as described below 
# and needs batch processing memory-efficient optimisation like 
# remove somata function has 

#adjusting only sigma itself gives the same output 
# maybe remove this function to estimate sigma or make it based on the width of 
# dendrites or make formula to find the best scale factor 


def tubenessForAllScenes(scenes, scale_factor=1.0):
    """
    Iterate over all scenes in a file, apply the tubeness function to each scene,
    and release memory after processing each scene.
    
    Parameters:
        scenes (list): List of 3D numpy arrays where each array represents a scene.
        scale_factor (float): Scaling factor for sigma estimation.
    
    Returns:
        list: A list of 3D numpy arrays with tubeness measure applied.
    """
    processed_scenes = []
    
    for i, scene in enumerate(scenes):
        print(f"Processing scene {i+1}/{len(scenes)}")
        
        if scene.size == 0:
            print(f"Scene {i+1} is empty or invalid!")
            continue
        
        try:
            sigma = estimate_sigma_based_on_scale(scene, scale_factor)
            processed_scene = tubeness(scene, sigma)
            processed_scenes.append(processed_scene)
            
        except Exception as e:
            print(f"Error processing scene {i+1}: {e}")
            continue
        
        del scene
        gc.collect()
    
    return processed_scenes

tubeness_scenes = tubenessForAllScenes(nosoma_scenes, scale_factor=0.9)

plot_images(tubeness_scenes[6][8,:,:], nosoma_scenes[6][8,:,:], 'Original', 'No Somata')

##################################### try to SAVE AS TIFF to see if it is only plotting bug 
############################################################################################
# Validation of tubeness 
import numpy as np
from skimage import exposure

def normalize_intensity_zero(image):
    """Normalize the intensity of a 3D image."""
    image_normalized = exposure.rescale_intensity(image, out_range=(0, 1))
    return image_normalized

def subtract_tubeness_from_nosoma(nosoma, tubeness):
    """
    Normalize the intensity of a single 3D nosoma image and a 3D tubeness image,
    and subtract the tubeness image from the nosoma image.
    
    Parameters:
        nosoma (ndarray): 3D numpy array with somas removed.
        tubeness (ndarray): 3D numpy array with tubeness measured.
    
    Returns:
        ndarray: A 3D numpy array after subtraction.
    """
    # Normalize both images
    nosoma_normalized = normalize_intensity_zero(nosoma)
    tubeness_normalized = normalize_intensity_zero(tubeness)
    
    # Subtract the tubeness image from the nosoma image
    result = nosoma_normalized - tubeness_normalized # swap to see if there are extra tubeness
    
    # Clip values to keep them in a valid range
    result = np.clip(result, 0, 1)
    
    # Release memory
    del nosoma, tubeness
    gc.collect()
    
    return result

# Example usage:
# Assuming `nosoma` and `tubeness` are 3D numpy arrays
validation = subtract_tubeness_from_nosoma(nosoma_scenes[8], tubeness_scenes[8])

plot_images(tubeness_scenes[8][8,:,:], nosoma_scenes[8][8,:,:], 'Original', 'No Somata')
plot_images(validation[8,:,:], nosoma_scenes[8][8,:,:], 'Original', 'No Somata')
#seems to work nicely, on specific images there are more "leftovers" 
# sometimes there is an extra tubeness 













# Example usage with a 3D numpy array `data`
# Sigma should be chosen based on the scale of the structures you're looking to enhance
sigma = 4.0  # Smoothing parameter 
result = tubeness(nosoma_scenes[8], sigma)
#plot_comparison(result[8,:,:], scenes[2][8,:,:], "Gaussian comparison") 
#or plot_images(blurred_result[8,:,:], blurred_scenes[2][8,:,:], 'Result Slice', 'Blurred Scene Slice')



#skeletonize (and project like in fiji)
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
#skeletonized = skeletonize_image(segmented)
#plot_images(blurred_result[8,:,:], skeletonized[8,:,:], 'Blurred result Slice', 'Skeletonized')
#print(skeletonized.shape)
#save_as_tiff(skeletonized, 'skeletonized.tif')
#save_as_tiff(scenes[2], 'scenes_2.tif')

################################
# tune parameters so the skeleton will be more accurate (some of the very low intensity or SMALL branches are not skeletonized)
# and clean skeleton afterwards + do labeling
################################

# validation of skeletonization 
# ..................

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

# Example: 

#mip_image = max_intensity_z_projection(skeleton)
#print(mip_image.shape)
#plot_images(blurred_result[8,:,:], mip_image, 'Blurred result Slice', 'Skeletonized')

#########################
# clean skeleton !!! remove small branches wit length smaller than ? (and weird very long??? )
# skeletonize try https://github.com/seung-lab/kimimaro

# i can label skeletons across all slices and then recognise where they intersect and where one 
# branch ends??? 

#label skeleton 
#measure branches 
# cut them or remove too short and what do i do with loo long branches 
##########################


###############################################################################
# fix clean skeleton function
# it should measure length of labeled skeleton and remove small objects 
# separate large objects? 

from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops
from skimage.io import imshow
from skimage.draw import line
from scipy.spatial.distance import euclidean

def measure_length(image_2d):
    """
    Calculate the Euclidean length of each branch in a skeletonized image.
    
    Args:
    image_2d (numpy.ndarray): The 2D binary image of skeletonized dendrites.
    
    Returns:
    list: List of lengths of each branch.
    """
    # Ensure the image is binary and skeletonized
    if image_2d.max() > 1:
        image_2d = image_2d > 0
    skeleton = skeletonize(image_2d)
    
    # Label connected components
    labeled_skeleton = label(skeleton)
    props = regionprops(labeled_skeleton)
    
    lengths = []
    for prop in props:
        # Extract the coordinates of the branch
        coords = prop.coords
        branch_length = 0
        # Calculate the Euclidean distance between consecutive pixels in the branch
        for i in range(len(coords) - 1):
            branch_length += euclidean(coords[i], coords[i + 1])
        lengths.append(branch_length)
    
    return lengths


measurments = measure_length(mip_image)


def process_dendrites(image_2d, min_length, max_length):
    """
    Process a 2D image of dendrites to measure and filter branches by length.
    
    Args:
    image_2d (numpy.ndarray): The 2D Z-projection image of dendrites, binary format.
    min_length (int): Minimum length of branches to keep.
    max_length (int): Maximum length of branches to keep.
    
    Returns:
    numpy.ndarray: The processed image with filtered dendritic branches.
    """
    # Ensure the image is binary and skeletonized
    if image_2d.max() > 1:
        image_2d = image_2d > 0
    skeleton = skeletonize(image_2d)
    
    # Label connected components
    labeled_skeleton = label(skeleton)
    
    # Measure properties of labeled regions
    props = regionprops(labeled_skeleton)
    
    # Filter out small and very long branches
    for prop in props:
        if prop.area < min_length or prop.area > max_length:
            for coord in prop.coords:
                skeleton[coord[0], coord[1]] = 0  # Set pixel to 0 (remove it)
    
    # Optionally, re-label to see the final branches
    filtered_skeleton = label(skeleton)
    
    return filtered_skeleton




# Example usage:
clean_skeleton = process_dendrites(mip_image, min_length=5, max_length=20000)

plot_images(mip_image, clean_skeleton, 'MIP', 'Clean')


#process to curliness



#Write image as tif. Ue imageJ for visualization
from skimage.io import imsave
imsave("tttt.tif", tubeness_scenes[6]) 

import tifffile as tiff

def save_as_tiff(image_slice, file_name):
    """Save an image slice as a TIFF file."""
    tiff.imwrite(file_name, image_slice, photometric='minisblack')

# Example usage to save specific slices
save_as_tiff(tubeness_scenes[6], 'tubeness_python.tif')
save_as_tiff(blurred_scenes[2][8, :, :], 'blurred_scene_slice_8.tif')





