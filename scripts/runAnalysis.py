#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 08:03:07 2024

@author: tetianasalamovska
"""

import sys
sys.path.append('/Users/tetianasalamovska/Documents/GitHub/nRIM_curliness_analysis') 

from src.FileImport.BatchProcessing import processFile, batchProcessFiles

# Example usage
folder_with_data = '/Users/tetianasalamovska/Desktop/zeis'
file_list = getMatchingFilesList(
    folder_with_data,
    EXPERIMENT='IHCT',
    MAGN_LIST=['40x2x', '40x3x'],
    ID='THT53',
    SEPARATOR='_',
    EXTENSION='.czi'
)


dataframe_results = batch_process_files(file_list, processFile, folder_with_data)
print(dataframe_results.head)
dataframe_results.to_csv('/Users/tetianasalamovska/Desktop/zeis/df.csv', index=False)

# Example: Get data for file_index 0, scene_index 2
filtered_df = df[(df['file_index'] == 0) & (df['scene_index'] == 2)]

# Example: Get all slices for a specific filename
specific_file_df = df[df['filename'] == 'example_filename.czi']

plot_images(dataframe_results[(dataframe_results['file_name'] == 'IHCT_THT53_40x2x_IHCT08_slice6_stack_positions_A488_laser08_speed6.czi') & (dataframe_results['scene_index'] == 6)], dataframe_results[(dataframe_results['file_name'] == 'IHCT_THT53_40x3x_IHCT08_slice6_stack_positions_A488_laser08_speed6.czi') & (dataframe_results['scene_index'] == 6)], 'Original', 'No Somata')

# Print summary of the DataFrame to check its content
print(dataframe_results.head())
print(dataframe_results['file_name'].unique())  # Check the unique filenames to ensure your file is listed
print(dataframe_results['scene_index'].unique())  # Check the unique scene indices
def fetch_and_plot_images(df, file_name_1, scene_index_1, file_name_2, scene_index_2):
    """
    Fetch two images based on file name and scene index, and plot them side by side.
    
    Args:
    df (DataFrame): The DataFrame containing the images and metadata.
    file_name_1 (str): File name of the first image.
    scene_index_1 (int): Scene index of the first image.
    file_name_2 (str): File name of the second image.
    scene_index_2 (int): Scene index of the second image.
    """
    image1_data = df.loc[(df['file_name'] == file_name_1) & (df['scene_index'] == scene_index_1), 'cleaned_scene'].iloc[0]
    image2_data = df.loc[(df['file_name'] == file_name_2) & (df['scene_index'] == scene_index_2), 'cleaned_scene'].iloc[0]
    
    if image1_data is not None and image2_data is not None:
        plot_images(image1_data, image2_data, 'Image 1', 'Image 2')
    else:
        print("One or both of the images could not be found.")

# Usage example
fetch_and_plot_images(dataframe_results, 
                      '/Users/tetianasalamovska/Desktop/zeis/IHCT_THT53_40x2x_IHCT08_slice6_stack_positions_A488_laser08_speed6.czi', 2, 
                      '/Users/tetianasalamovska/Desktop/zeis/IHCT_THT53_40x2x_IHCT08_slice6_stack_positions_A488_laser08_speed6.czi', 9)


