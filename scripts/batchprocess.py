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
import matplotlib.pyplot as plt
from src.FileImport.DesctiptorsBasedFileSearch import getMatchingFilesList
#from src.FileImport.BatchProcessing import 
from src.FileImport.ReadZeissStacks import readCziFile
from src.ImageProcessing.NormilizeIntensity import normalizeScenes
from src.FileImport.PlottingImage import plotToCompare, plotImageHistogram
from src.ImageProcessing.Skeletonize import applyGaussian #change to new file 
from src.ImageProcessing.SubstractBackground import subtractBackgroundFromScenes
#from src.ImageProcessing.SatoTubeness import applySatoTubeness 




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

# Step 1. Importing files that match file list names (here just importing one file) and reading metadata
    # 1.1. Read the file as a list of 3D numpy arrays ( or "scenes" like in your single file, there are 11 of them)

file_name = '/Users/tetianasalamovska/Desktop/zeis/IHCT_THT53_40x3x_IHCT08_slice6_stack_positions_A488_laser08_speed6.czi'
scenes, metadata = readCziFile(file_name)

    # 1.2. Create a metadata dictionary and access it with this function 







# Print the shape of scenes just to check 
#if scenes:
    #print(scenes[0-10].shape)


# Step 2. Preprocessing

    # 2.1 Normalizing intensities and enhancing contrast
    
# Also returnes normalized image shape and min and max intensities in the console
# You can print it using print(np.min(scenes[5]))
normalized_scenes = normalizeScenes(scenes, percentiles=[0.1,99.9])


# Optionally visualize one of the slices in the one of the stacks before and after 
# Here it is important to remember image shape structure: scenes - your list, [4] - index of scenes,
# in Python index starts from 0, [8,:,:] - this means 8 slice with all X and Y dimension values
plotToCompare(scenes[6][10,:,:], normalized_scenes[6][10,:,:], 'Original', 'Normalized')

# Inspect histograms if needed 
plotImageHistogram(normalized_scenes[6], bins=256, pixel_range=(0, 65535), title='Pixel Intensity Histogram for Normalized Image')

# del scenes

    # 2.2. Denoising and optional morphological techniques
blurred_scenes = applyGaussian(normalized_scenes, sigma=2)

plotToCompare(normalized_scenes[6][10,:,:], blurred_scenes[6][10,:,:], 'Normalized', 'Gaussian Blur')

    # 2.3. Background substraction, try radius from 25 to 35++
    
#for one stack to test different radius 
#subtracted_scene = subtract_background(blurred_scenes[6], radius=35) # 20 is also very fine 

subtracted_scenes = subtractBackgroundFromScenes(scenes, radius=25)

plotToCompare(subtracted_scenes[7][10,:,:], blurred_scenes[7][10,:,:], 'Substracted', 'Gaussian Blur')

# Might need some contrast enhancement here !!!!!!!!!!!!!!!!


    # Do not use: 2.4. Tubeness filter  (might need more preprocessing before this, more filters or opening/closing operations
    # or pref. demoving some small objects, not noise + contrast enhancement)

# sato tubeness from scikit image 
#tubeness_test = applySatoTubeness(subtracted_scenes[7], sigma = 1, black_ridges=False)
#plotToCompare(subtracted_scenes[7][10,:,:], tubeness_test[10,:,:], 'Substracted', 'Tubeness')


    # 2.5. Contrast enhancement, small obj. cleaning, similar to tubeness morphological operation


from skimage.morphology import white_tophat, ball

selem = ball(radius=3)
enhanced_image = white_tophat(subtracted_scenes[7], footprint=selem)

plotToCompare(subtracted_scenes[7][10,:,:], enhanced_image[10,:,:], 'Substracted', 'Filter')


# median? constrast (stretching? Adaptive Histogram Equalization?), opening, closing? clean? 



# Step 3. Thresholding and binarisation












