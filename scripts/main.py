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
sigmaX = 3      # Standard deviation

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
plot_comparison(adjusted_scenes[5][8,:,:], blurred_scenes[5][8,:,:], "Background Correction Comparison")


