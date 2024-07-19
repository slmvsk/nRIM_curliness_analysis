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
from src.FileImport.PlottingImage import plot_slice_from_stack
from src.ImageProcessing.NormilizeIntensity import normalize_intensity, validate_image_adjustment, process_scenes

import sys
sys.path.append('/Users/tetianasalamovska/Documents/GitHub/nRIM_curliness_analysis')

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


print(blurred_scenes[2].shape)
print(adjusted_scenes[2].shape)

# Example of plotting comparison for the first scene
plot_comparison(adjusted_scenes[1][8,:,:], blurred_scenes[1][8,:,:], "Gaussian comparison")

#tubness method adapted from imagej 

import numpy as np
from skimage.filters import gaussian
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from src.ImageProcessing.ImageProcessing import enhance_neurites

# Example usage (assuming you have an image 'img' and a sigma value)
enhanced_scenes = enhance_neurites(blurred_scenes[1], sigma=6)
plot_comparison(adjusted_scenes[1][1,:,:], blurred_scenes[1][1,:,:], "Tubeness comparison")


#enhanced_image = meijering(blurred_scenes[1], sigmas=range(1, 10, 2), black_ridges=True)

plot_comparison(adjusted_scenes[1][1,:,:], blurred_scenes[1][1,:,:], "Comparison")

import imagej
import os

# Initialize ImageJ
ij = imagej.init('sc.fiji:fiji')  # Make sure Fiji is installed correctly and the path is recognized

# Assuming 'scenes' is your list of 3D numpy arrays
scenes = [np.random.random((100, 100, 100)) for _ in range(5)]  # Example scenes

# List to hold the processed scenes
processed_scenes = []

# Iterate over each scene
for i, scene in enumerate(scenes):
    # Convert the numpy array to an ImageJ2 compatible image
    image = ij.py.to_java(scene)

    # Apply the Tubeness plugin
    # Adjust 'sigma' and other parameters according to your needs
    ij.py.run_plugin('FeatureJ', {
        'command': 'Tubeness',
        'sigmas': [1.0],  # List of sigmas to process with
        'output': ['Tubeness'],
        'input': image
    })

    # Retrieve the processed image
    processed_image = ij.py.from_java(ij.py.get_image_plus())  # Converts the result back to a numpy array

    # Append the result to the list of processed scenes
    processed_scenes.append(processed_image)

    print(f"Processed scene {i + 1}/{len(scenes)}")

# Cleanup ImageJ resources
ij.dispose()

# Optionally, display or analyze the processed scenes further
print("Processing complete. Processed scenes are stored in the 'processed_scenes' list.")

