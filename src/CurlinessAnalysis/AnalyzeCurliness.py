#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:16:21 2024

@author: tetianasalamovska
 
"""

# Input image is max intensity Z-projection of skeletons 
# mip_image


# measurements: fiber length density? francal nature(dF - fractal dimensions (berween 1 and 2
# ))

# Straightness = max. Dendritic reach / longest single path length 
# max. dendritic reach is the shortest path between the start and end points of the branch
# longest single path length is  the actual path length measured along the branch, 
# which will be longer than the branch distance for any curved path! 

# Curliness will be 1 - straightness that implies that a perfectly straight branch (for ex. 15/15 = 1)
# will have curliness equal 0 (1-1 = 0). More curved branches have higher values approaching 1

# clean skeleton before applying or you get minus curliness because your shortest path between the start and end points of the branch
#is too long and you get straightness more than 1 in some cases. They are just connected and 
# algorithm sees them as 1 branch 

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, io, morphology
from scipy.ndimage import distance_transform_edt
import numpy as np
from skimage import measure

def analyze_dendrite_curliness(image):
    # Label the skeleton
    labeled_skeleton = measure.label(image)
    properties = measure.regionprops(labeled_skeleton)

    longest_path_length = []
    max_dendritic_reach = []

    # Calculate measures
    for prop in properties:
        # longest_path_length 
        longest_path_length.append(prop.area)  # Assuming area as a proxy for path length
        
        # Calculate straight-line (Euclidean) distance between the end points of the skeleton
        minr, minc, maxr, maxc = prop.bbox
        distance = np.sqrt((maxr - minr) ** 2 + (maxc - minc) ** 2)
        max_dendritic_reach.append(distance)

    # Output each branch's measures
    for length, reach in zip(longest_path_length, max_dendritic_reach):
        print(f"Branch: Length = {length}, Max Reach = {reach}")

    # Convert lists to arrays for numerical operations
    longest_path_length = np.array(longest_path_length)
    max_dendritic_reach = np.array(max_dendritic_reach)

    # Calculate straightness and curliness
    straightness = max_dendritic_reach / longest_path_length
    curliness = 1 - straightness

    # Calculate average, std, and sem of curliness
    mean_straightness = np.mean(straightness)
    mean_curliness = np.mean(curliness)
    std_curliness = np.std(curliness)
    median_curliness = np.median(curliness)
    sem_curliness = std_curliness / np.sqrt(len(longest_path_length))

    return curliness, median_curliness, mean_straightness, mean_curliness, sem_curliness, longest_path_length.tolist(), max_dendritic_reach.tolist()



# Example usage
curliness, median_curliness, mean_straightness, mean_curliness, sem_curliness, branch_distances, branch_lengths = analyze_dendrite_curliness(mip_image)
# Retrieve the image data for scene index 2
# We assume that 'file_name' or another identifier may be needed if there are multiple entries for the same scene index.
# Here, I'm directly accessing by index if the scene index is used as a row index. If not, you'd filter by conditions.
image_data = dataframe_results[dataframe_results['scene_index'] == 2]['cleaned_scene'].iloc[0]

# Now apply the analyze function
mean_straightness, mean_curliness, sem_curliness, longest_path_length, max_dendritic_reach = analyze_dendrite_curliness(image_data)
print("Mean Straightness:", mean_straightness)
print("Mean Curliness:", median_curliness)
print("Longest Path Length:", longest_path_length)
print("Maximum Dendritic Reach:", max_dendritic_reach)


# Plotting longest_path_length
plt.figure(figsize=(6, 4))
plt.hist(curliness, bins=100, color='blue', alpha=0.7)
plt.title('Histogram of longest_path_length')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.xlim([0, 1])  # Set x-axis limits
plt.ylim([0, 6])  # Optionally adjust the y-axis to change how density appears
plt.show()

# Plotting max_dendritic_reach
plt.figure(figsize=(6, 4))
plt.hist(branch_distances, bins=10000, color='green', alpha=0.7)
plt.title('Histogram of max_dendritic_reach')
plt.xlabel('max_reach')
plt.ylabel('Frequency')
plt.xlim([0, 300])  # Set x-axis limits for example 
plt.ylim([0, 50])  # Optionally adjust the y-axis to change how density appears
plt.show()

# Plotting mean curliness across groups 

############################# dataframe is an input 

import numpy as np
import pandas as pd
from skimage import measure

def analyze_dendrite_curliness_batch(dataframe):
    # Initialize lists to hold results
    mean_straightness = []
    mean_curliness = []
    sem_curliness = []
    file_names = []
    scene_indices = []

    # Process each image in the dataframe
    for index, row in dataframe.iterrows():
        image = row['cleaned_scene']
        labeled_skeleton = measure.label(image)
        properties = measure.regionprops(labeled_skeleton)

        longest_path_length = []
        max_dendritic_reach = []

        for prop in properties:
            longest_path_length.append(prop.area)
            minr, minc, maxr, maxc = prop.bbox
            distance = np.sqrt((maxr - minr) ** 2 + (maxc - minc) ** 2)
            max_dendritic_reach.append(distance)

        longest_path_length = np.array(longest_path_length)
        max_dendritic_reach = np.array(max_dendritic_reach)
        straightness = max_dendritic_reach / longest_path_length
        curliness = 1 - straightness

        mean_straightness.append(np.mean(straightness))
        mean_curliness.append(np.mean(curliness))
        std_curliness = np.std(curliness)
        sem_curliness.append(std_curliness / np.sqrt(len(longest_path_length)))
        file_names.append(row['file_name'])
        scene_indices.append(row['scene_index'])

    # Create a new DataFrame with the results
    results_df = pd.DataFrame({
        'file_name': file_names,
        'scene_index': scene_indices,
        'mean_straightness': mean_straightness,
        'mean_curliness': mean_curliness,
        'sem_curliness': sem_curliness
    })

    return results_df

# Assuming 'dataframe_results' is your DataFrame that contains all the processed images.
curliness_df = analyze_dendrite_curliness_batch(dataframe_results)
print(curliness_df.head())
curliness_df.to_csv('/Users/tetianasalamovska/Desktop/zeis/curliness_df.csv', index=False)

########### Box counting method is the best measure (fractality)
# look into curliness (how it identifies branches......)
# try lkshm code with distance matrix 
# label skeletons to see how it identifies dendrites 
 











