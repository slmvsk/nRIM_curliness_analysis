#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 14:29:08 2024

@author: tetianasalamovska
"""
#IN PROGRESS (NOT A FUNCTION)


# •	Pandas DataFrame: For managing and comparing large datasets efficiently, consider using a pandas DataFrame. This can allow you to quickly access metadata and link scenes by index.

# using dictionaries and file handling 
#this is the first thing to do because i need to adapt functions for batch processing
# and also be able to compare different images in the end 
# i need image list that witll have index name type size value so basically list of numpy arrays like I have 
# 1 scene image but for a lot of images, so indexes will go up, like my created df
# and then I need to convert list to numpy array 
# and then we process each "slice" in the stack?? using indexes

    
from czitools.metadata_tools import czi_metadata as czimd
import numpy as np
from czifile import CziFile
import os
import pandas as pd


# update this function in its file 
def readCziFile(file_path):
    """
    Reads a ZEISS CZI file and returns a list of 3D numpy arrays (one per scene) 
    and metadata.
    
    Parameters:
    - file_path (str): Path to the CZI file
    
    Returns:
    - scenes (list of numpy arrays): List of 3D numpy arrays representing image data for each scene (z-stack)
    - metadata (dict): Metadata associated with the CZI file
    """
    with CziFile(file_path) as czi:
        
         # Get the dimensions of the CZI file
        czi_dimensions = czimd.CziDimensions(file_path)
        
        num_scenes = czi_dimensions.SizeS if czi_dimensions.SizeS else 1
        #Convert the entire image to a numpy array
        full_image = czi.asarray()
        # Determine the shape of the full image array
        print("Full image shape: ", full_image.shape)
        
        # Initialize list to store 3D numpy arrays for each scene
        scenes = []
        # allscenes = metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['S']['Scenes']['Scene']

        # Extract scenes assuming shape (1, 1, S, 1, 1, Z, Y, X, 1)
        for scene_idx in range(num_scenes):
            scene_data = full_image[0, 0, scene_idx, 0, 0, :, :, :, 0]
            scenes.append(scene_data)
        
        # Extract metadata using czitools
        metadata = czimd.CziMetadata(file_path) #all metadata
        

    return scenes, metadata 

scenes, metadata = readCziFile('/Users/tetianasalamovska/Desktop/zeis/IHCT_THT53_40x3x_IHCT08_slice6_stack_positions_A488_laser08_speed6.czi')
print(metadata)
# so my base function returnes scenes as list of 3D numpy arrays and metadata (all metadata for all scenes)
# my next function should iterate this function for all images in folder that match file names that are stored in "file_list"
# and organize to store all of theseas indexed images in dataframe from which i can call specific slices (for example image with index 1, scene index 10, slice index
# 19) and I want also into this dataframe incude some specific metadata extracted from all metadata as additional columns 






metadata_dict=extract_metadata(metadata)


print(f"Type of metadata: {type(metadata)}")
print(f"Metadata content (string): {metadata}")
print(f"Metadata content: {metadata}")


# WRITE A FUNCTION TO CREATE METADATA DICT LATER 
# RIGHT NOW CONCENTRATE ON CREATING INDEXED DF WITHOUT METADATA TO ANALYSE FEW IMAGES AT THE SAME TIME
# AND LEARN HOW TO CALL SLICES FROM DF AND HOW TO HANDLE IT USING "FILE_NAMES"
# HERE 







import os
import pandas as pd

def createDataframeFromFileList(folder_path, file_list):
    records = []

    for filename in file_list:
        czi_file_path = os.path.join(folder_path, filename)
        scenes, metadata = readCziFile(czi_file_path)  # Ignore metadata for this function

        for scene_idx, scene_data in enumerate(scenes):
            for slice_idx in range(scene_data.shape[0]):  # Assuming scene_data is 3D with shape (20, height, width)
                records.append([filename, 0, scene_idx, slice_idx, scene_data[slice_idx, :, :]])
    
    df = pd.DataFrame(records)
    return df


folder_path = '/Users/tetianasalamovska/Desktop/zeis'
# file_list from function that finds files 

# Create the DataFrame
df = createDataframeFromFileList(folder_path, file_list)

# Save the DataFrame to CSV
csv_file = '/Users/tetianasalamovska/Desktop/zeis/file.csv'
df.to_csv(csv_file, index=False)

print(f"DataFrame saved to {csv_file}") # indexes correctly 



# Example: Get data for file_index 0, scene_index 2
filtered_df = df[(df['file_index'] == 0) & (df['scene_index'] == 2)]

# Example: Get all slices for a specific filename
specific_file_df = df[df['filename'] == 'example_filename.czi']

# Access specific slice data
slice_data = filtered_df[filtered_df['slice_index'] == 5]['scene_data'].values[0]  # Get 2D data for slice 5



# iterating example
for filename in df['filename'].unique():
    file_data = df[df['filename'] == filename]
    for scene_index in file_data['scene_index'].unique():
        scene_data = file_data[file_data['scene_index'] == scene_index]
        for slice_index in scene_data['slice_index']:
            slice_data = scene_data[scene_data['slice_index'] == slice_index]['scene_data'].values[0]
            # Process the slice_data





################## from other function (to be imported)

# Call the function without any descriptors (will return all files in the folder)
files_list = get_matching_files(folder_with_data)
print(file_list)


# Example: Adjust parameters based on your file naming convention
file_list = get_matching_files(folder_with_data, EXPERIMENT='IHCT', MAGN=[40x3x, 40x2x], ID='THT53', SEPARATOR='_')
print("Matching Files:", file_list)


# Usage
directory = '/Users/tetianasalamovska/Desktop/zeis'
data_index = load_images(directory, file_list)
df = create_dataframe(data_index)
print("hi")
print(df.info())  # Check the first few rows of the DataFrame

csv_file = '/Users/tetianasalamovska/Desktop/zeis/file.csv'
df.to_csv(csv_file, index=False)





    
    


