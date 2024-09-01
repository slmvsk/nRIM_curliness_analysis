#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 14:33:32 2024

@author: tetianasalamovska
"""

# Git clone repository in terminal (instructions in README.txt file)
# Navigate to the directory: this will be replaced with real path on cluster/your computer 

import sys
sys.path.append('/Users/tetianasalamovska/Documents/GitHub/nRIM_curliness_analysis') 


# Importing all related functions after navigating to downloaded repository 
# Also might need requirments.txt file 

import numpy as np
from src.FileImport.DesctiptorsBasedFileSearch import getMatchingFilesList
#from src.FileImport.BatchProcessing import 
from src.FileImport.ReadZeissStacks import readCziFile
from src.ImageProcessing.NormilizeIntensity import normalizeScenes
from src.FileImport.PlottingImage import plotToCompare







# Choosing files you want to analyze, assuming they are all in one folder 
# Define descriptors that you need or just run function without descriptors to get all file names in the folder 

folder_with_data = '/Users/tetianasalamovska/Desktop/zeis'
file_list = getMatchingFilesList(
    folder_with_data,
    EXPERIMENT='IHCT',
    MAGN_LIST=['40x3x'], # you can mention a few ['40x2x', '40x3x'] or leave empty '' for function to ignore it 
    ID='THT53',
    SEPARATOR='_',
    EXTENSION='.czi')

#print("Matching Files:", file_list)


# !!! There must be a batch process function, but I will write a test analysis for 1 file instead 
# and then just put it in the function at src.FileImport.BarchProcessing 

# Importing files that match file list names (here just importing one file)

file_name = '/Users/tetianasalamovska/Desktop/zeis/IHCT_THT53_40x3x_IHCT08_slice6_stack_positions_A488_laser08_speed6.czi'
scenes, metadata = readCziFile(file_name)

# Create a metadata dictionary and access it with this function 







# Print the shape of scenes just to check 
#if scenes:
    #print(scenes[0-10].shape)


# Normalizing intensities and enhancing contrast
# Also returnes normalized image shape and min and max intensities in the console
# You can print it using print(np.min(scenes[5]))
normalized_scenes = normalizeScenes(scenes, percentiles=[0.1,99.9])


# Optionally visualize one of the slices in the one of the stacks before and after 
plotToCompare(scenes[4][8,:,:], normalized_scenes[8,:,:], 'Original', 'Normalized')


# Inspect histograms if needed 





















