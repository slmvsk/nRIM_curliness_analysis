#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 09:33:53 2024

@author: tetianasalamovska
"""

# main.py
from czitools.metadata_tools import czi_metadata as czimd
import numpy as np
from czifile import CziFile
import pyclesperanto_prototype as cle

import sys
sys.path.append('/Users/tetianasalamovska/Documents/GitHub/nRIM_curliness_analysis')


from src.FileImport.PlottingImage import plot_slice_from_stack, plot_comparison
from src.ImageProcessing.NormilizeIntensity import normalize_intensity, validate_image_adjustment, process_scenes

from src.FileImport.ReadZeissStacks import read_czi_stack

file_path = '/Users/tetianasalamovska/Desktop/zeis/IHCT_THT53_40x2x_IHCT08_slice6_stack_positions_A488_laser08_speed6.czi'

scenes, metadata = read_czi_stack(file_path)

# Print the metadata
#prints all metadata (no need)
print(metadata)

# Print the shape of the first scene (all)
if scenes:
    print(scenes[0-10].shape)

#print selected metadata ??? 
czi_scaling = czimd.CziScaling(file_path)
czi_channels = czimd.CziChannelInfo(file_path)
czi_bbox = czimd.CziBoundingBox(file_path)
czi_objectives = czimd.CziObjectives(file_path)
czi_detectors = czimd.CziDetector(file_path)
czi_microscope = czimd.CziMicroscope(file_path)
czi_sample = czimd.CziSampleInfo(file_path)

#centerposX = []
#centerposY = []

# normalizing intensities min 0 max 65535 for 16 bit 
adjusted_scenes = process_scenes(scenes)

if adjusted_scenes:
    print(adjusted_scenes[0-10].shape)

# plot to compare original and normalized image 
# Example usage assuming 'scenes' and 'adjusted_scenes' are your lists of 3D numpy arrays
slice_index = 5  # Specify the slice index you want to compare
if slice_index < len(scenes[1]) and slice_index < len(adjusted_scenes[1]):
    plot_slice_from_stack(scenes[1], slice_index)
    plot_slice_from_stack(adjusted_scenes[1], slice_index)
else:
    print("Specified slice index is out of range.")

# remove somatas function apply (finish writing)

#from src.ImageProcessing.ImageProcessing import gaussian_filter
import cv2

# Parameters for Gaussian blur
ksize = (5, 5)  # Kernel size should be odd numbers
sigmaX = 4      # Standard deviation

# Apply Gaussian blur to each slice of each 3D array
blurred_scenes = []
for scene in adjusted_scenes:
    blurred_scene = np.empty_like(scene)  # Create an empty array to store the blurred slices
    for z in range(scene.shape[0]):
        blurred_scene[z] = cv2.GaussianBlur(scene[z], ksize, sigmaX)  # Apply blur to each slice
    blurred_scenes.append(blurred_scene)

# Optionally, check the output
for index, scene in enumerate(blurred_scenes):
    print(f"Blurred scene {index} shape: {scene.shape}")


if blurred_scenes:
    print(blurred_scenes[0-10].shape)

print(blurred_scenes[2].shape)
print(adjusted_scenes[2].shape)

# Example of plotting comparison for the first scene
plot_comparison(adjusted_scenes[1][8,:,:], blurred_scenes[1][8,:,:], "Gaussian comparison")

sigma = 4.0  # Gaussian smoothing parameter
result = tubeness(blurred_scenes[2], sigma)

plot_comparison(result[8,:,:], blurred_scenes[2][8,:,:])

print(result.shape)

from scipy.ndimage import gaussian_filter
blurred_result = gaussian_filter(result, sigma=5)

plot_comparison(blurred_result[8,:,:], blurred_scenes[2][8,:,:], "Plot comparison")

#median filter can be better alternative to gaussian blur 
medianfilter_image = scipy.ndimage.median_filter(scenes[1], size=5)

#print(medianfilter_image.shape)

plot_comparison(scenes[1][8,:,:], medianfilter_image[8,:,:], "Filter comparison")

#Tubness (includes Gaussian smoothing filter )
sigma = 5.0  # Smoothing parameter 
result = tubeness(image_nosoma, sigma)

plot_comparison(result[8,:,:], scenes[2][8,:,:], "Gaussian comparison")

#another way to plot in better quality (only 2D selected)
plot_images(blurred_result[8,:,:], blurred_scenes[2][8,:,:], 'Result Slice', 'Blurred Scene Slice')


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

# Example usage:
# Assume `image_3d` is your 3D numpy array that's already a binary image
skeletonized = skeletonize_image(segmented)

plot_images(blurred_result[8,:,:], skeletonized[8,:,:], 'Blurred result Slice', 'Skeletonized')

print(skeletonized.shape)
save_as_tiff(skeletonized, 'skeletonized.tif')
save_as_tiff(scenes[2], 'scenes_2.tif')
# tune parameters so the skeleton will be more accurate (some of the very low intensity or SMALL branches are not skeletonized)

# validation 
#......

mip_image = max_intensity_z_projection(skeletonized)

print(mip_image.shape)
plot_images(blurred_result[8,:,:], mip_image, 'Blurred result Slice', 'Skeletonized')










