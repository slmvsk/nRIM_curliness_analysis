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


medianfilter_image = scipy.ndimage.median_filter(scenes[1], size=5)

#print(medianfilter_image.shape)

plot_comparison(scenes[1][8,:,:], medianfilter_image[8,:,:], "Filter comparison")

##########################
# Tubeness problem:
# it wants to see image with only tubular structures, so sometimes it connects dots
# Solution: 
# Preprocess better, dots and small onjects doesn't give me any information (dendrite pieces)
##########################

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
sigma = 5.0  # Gaussian smoothing parameter
result = tubeness(image_nosoma, sigma)

plot_comparison(result[8,:,:], scenes[2][8,:,:], "Gaussian comparison")


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
backgrund_subtracted = cle.top_hat_box(result, radius_x=10, radius_y=10, radius_z=10)
print(result.shape)

print(backgrund_subtracted.shape)
#not bad but radiuses to be chosen and maybe another method

plot_images(result[8,:,:], backgrund_subtracted[8,:,:], 'Blurred result Slice', 'Subtrackted background')


# problems with segmentation 
segmented = cle.voronoi_otsu_labeling(backgrund_subtracted, spot_sigma=3, outline_sigma=1)

print(segmented.shape)

plot_images(result[8,:,:], segmented[8,:,:], 'Blurred result Slice', 'Segmented')


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
# tune parameters so the skeleton will be more accurate (some of the very low intensity or SMALL branches are not skeletonized)

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

# clean skeleton !!! remove small branches wit length smaller than ? (and weird very long??? )
# skeletonize try https://github.com/seung-lab/kimimaro

# i can label skeletons across all slices and then recognise where they intersect and where one 
# branch ends??? 

#label skeleton 
#measure branches 
# cut them or remove too short and what do i do with loo long branches 








# fix clean skeleton function 
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





