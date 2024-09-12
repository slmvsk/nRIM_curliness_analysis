#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 14:29:08 2024

@author: tetianasalamovska
"""

import scipy
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from src.FileImport.DesctiptorsBasedFileSearch import getMatchingFilesList
#from src.FileImport.BatchProcessing import 
from src.FileImport.ReadZeissStacks import readCziFile
from src.ImageProcessing.NormilizeIntensity import normalizeScenes
from src.FileImport.PlottingImage import plotToCompare, plotImageHistogram, visualize3dMayavi
from src.ImageProcessing.DenoisingFilters import applyGaussian, applyMedianFilter, applyContrastStretching
from src.ImageProcessing.SubstractBackground import subtractBackgroundFromScenes
#from src.ImageProcessing.SatoTubeness import applySatoTubeness 
from src.ImageProcessing.Binarize import removeSomaFromAllScenes, cleanBinaryScenes
from src.ImageProcessing.Skeletonize import skeletonizeScenes, pruneScenes, zProjectScenes, cleanMipSkeleton, prune3Dscenes,removeLoopsScenes, breakJunctionsAndLabelScenes
from src.ImageProcessing.Morphology import applyErosionToScenes, applyDilationToScenes
#from src.CurlinessAnalysis.AnalyzeCurliness import analyzeCurlinessBatch
from src.ImageProcessing.Thresholding import otsuThresholdingScenes


def processFile(file_name):
    # Step 1: Read the file as a list of 3D numpy arrays (scenes)
    scenes, metadata = readCziFile(file_name)
    
    # Step 2: Normalize intensity 
    normalized_scenes = normalizeScenes(scenes, percentiles=[1,99])
    del scenes  # Free memory used by the original scenes
    
    # Step 3: Denoising 
    blurred_scenes = applyGaussian(normalized_scenes, sigma=2)
    del normalized_scenes
    # Step 4: Background substraction and contrast enhancement 
    subtracted_scenes = subtractBackgroundFromScenes(blurred_scenes, radius=25)
    del blurred_scenes
    
    median_scenes = applyMedianFilter(subtracted_scenes, size=3)
    del subtracted_scenes
    
    stretched_scenes = applyContrastStretching(median_scenes, lower_percentile=1, upper_percentile=99)
    del median_scenes
    
    # Step 5: Remove soma
    binary_scenes = otsuThresholdingScenes(stretched_scenes) # maybe put old nosoma adaptive thresholding function back here 
    del stretched_scenes
    
    # Step 6: Cleaning
    cleaned_scenes = cleanBinaryScenes(binary_scenes, min_size=4000) 
    del binary_scenes
    
    eroded_scenes = applyErosionToScenes(cleaned_scenes, iterations=2, structure=np.ones((3, 3, 3)))
    del cleaned_scenes
    
    dilated_scenes = applyDilationToScenes(eroded_scenes, iterations=2, structure=np.ones((3, 3, 3)))  
    del eroded_scenes
    
    # Step 5: Skeletonize and clean, prune skeleton 
    skeletonized_scenes = skeletonizeScenes(dilated_scenes)
    del dilated_scenes
    
    pruned_scenes3D = prune3Dscenes(skeletonized_scenes, size=30) #here removes small branches in 3d as well
    del skeletonized_scenes
    
    z_projected_scenes = zProjectScenes(pruned_scenes3D) # shifting to 2D image 
    del pruned_scenes3D
    
    cleaned_2d_skeletons = cleanMipSkeleton(z_projected_scenes, min_length=100, max_length=30000) #this 
    
    pruned_scenes, segmented_scenes, segment_objects_list = pruneScenes(cleaned_2d_skeletons, size=30, mask=None) # side branches removal 
    
    skeletonizedscenes = removeLoopsScenes(pruned_scenes)
    del pruned_scenes
    
    # More pruning if needed 
    # pruned_scenes, segmented_scenes, segment_objects_list = pruneScenes(cleaned_2d_skeletons, size=30, mask=None) 

    output_skeletons = breakJunctionsAndLabelScenes(skeletonizedscenes, num_iterations=3)
    del skeletonizedscenes
    
    print(f"Finished processing file: {file_name}")
    
    return output_skeletons



def batchProcessFiles(file_list, process_function, folder_path):
    """
    Batch processes files and compiles results into a pandas DataFrame.
    
    Args:
    file_list (list): List of file names to be processed.
    process_function (function): Function to process each file.
    folder_path (str): Path to the folder containing the files.
    
    Returns:
    pd.DataFrame: DataFrame containing results from all processed files.
    """
    # List to store DataFrame rows before concatenation
    all_rows = []

    for file_name in file_list:
        # Construct full file path
        file_path = os.path.join(folder_path, file_name)
        
        # Process the file
        output_skeletons = processFile(file_path)
        
        # Collect each scene's data in all_rows list
        for scene_index, cleaned_scene in enumerate(cleaned_scenes):
            all_rows.append({
                "file_name": file_name,
                "scene_index": scene_index,
                "output_skeletons": output_skeletons
            })
        
        print(f"Finished processing {file_name}")
    
    # Convert list of dicts to DataFrame
    results_df = pd.DataFrame(all_rows)
    
    return results_df



def release_memory(variable):
    del variable
