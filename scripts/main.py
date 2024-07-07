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
from src.ImageProcessing.NormilizeIntensity import normalize_intensity, validate_image_adjustment, process_scenes

adjusted_scenes = process_scenes(scenes)

if adjusted_scenes:
    print(adjusted_scenes[0-10].shape)

# plot to compare original and normalized image 


