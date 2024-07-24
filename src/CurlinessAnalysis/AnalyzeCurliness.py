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
# which will be longer than the branch distance for any curved path.


import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, io, morphology
from scipy.ndimage import distance_transform_edt

def analyze_dendrite_curliness(image):

    # Label the skeleton
    labeled_skeleton = measure.label(image)
    properties = measure.regionprops(labeled_skeleton)

    longest_path_length = []
    max_dendritic_reach = []

    # Calculate measures
    for prop in properties:
        # longest_path_length 
        longest_path_length.append(prop.area)
        
        # Calculate straight-line (Euclidean) distance between the end points of the skeleton
        minr, minc, maxr, maxc = prop.bbox
        distance = np.sqrt((maxr - minr) ** 2 + (maxc - minc) ** 2)
        max_dendritic_reach.append(distance)

    # Straightness 
    longest_path_length = np.array(longest_path_length)
    max_dendritic_reach = np.array(max_dendritic_reach)
    straightness = max_dendritic_reach / longest_path_length
    # Straightness = max. Dendritic reach / longest single path length 
    # Curliness inverse 
    curliness = 1 - straightness

    # Output curliness measures
    mean_straightness = np.mean(straightness)
    mean_curliness = np.mean(curliness)
    std_curliness = np.std(curliness)
    sem_curliness = std_curliness / np.sqrt(len(longest_path_length))

    return mean_straightness, mean_curliness, sem_curliness, longest_path_length, max_dendritic_reach


# Example usage
mean_straightness, mean_curliness, sem_curliness, branch_distances, branch_lengths = analyze_dendrite_curliness(mip_image)
print(f"Mean Straightness: {mean_straightness}")

# Plotting longest_path_length
plt.figure(figsize=(6, 4))
plt.hist(branch_lengths, bins=100, color='blue', alpha=0.7)
plt.title('Histogram of longest_path_length')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.xlim([0, 1000])  # Set x-axis limits
plt.ylim([0, 150])  # Optionally adjust the y-axis to change how density appears
plt.show()

# Plotting branch distances
plt.figure(figsize=(6, 4))
plt.hist(branch_distances, bins=100, color='green', alpha=0.7)
plt.title('Histogram of max_dendritic_reach')
plt.xlabel('max_reach')
plt.ylabel('Frequency')
plt.xlim([0, 300])  # Set x-axis limits for example 
plt.ylim([0, 150])  # Optionally adjust the y-axis to change how density appears
plt.show()



