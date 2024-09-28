#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 12:55:55 2024

@author: tetianasalamovska


Analysis inspired by https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5439180/
They have their matlab code but I didn't look into it because describing the logic to gpt worked out

"""


import numpy as np
import os
import sys
import pandas as pd
sys.path.append('/Users/tetianasalamovska/Documents/GitHub/nRIM_curliness_analysis') 
from src.FileImport.BatchFunction import processFileforFrac



def boxCount(binary_image, min_box_size=2):
    
    """ Calculate Box Sizes: Use powers of 2 to define box sizes.
    	Loop Over Box Sizes: For each box size, divide the image into boxes.
        Count Non-Empty Boxes: For each box, check if it contains any part of the dendrite.
        Collect Counts: Store the number of non-empty boxes for each box size. """
    # Get the size of the image
    rows, cols = binary_image.shape

    # Maximum box size is the size of the image
    n = min(rows, cols)
    sizes = 2 ** np.arange(np.floor(np.log2(n)), np.floor(np.log2(min_box_size)) - 1, -1).astype(int)

    counts = []
    for size in sizes:
        # Number of boxes along each dimension
        n_rows = int(np.ceil(rows / size))
        n_cols = int(np.ceil(cols / size))

        # Count the number of non-empty boxes
        count = 0
        for i in range(n_rows):
            for j in range(n_cols):
                # Define the box boundaries
                row_start = i * size
                row_end = min((i + 1) * size, rows)
                col_start = j * size
                col_end = min((j + 1) * size, cols)

                # Extract the box
                box = binary_image[row_start:row_end, col_start:col_end]

                # Check if the box contains any foreground pixels
                if np.any(box):
                    count += 1

        counts.append(count)

    return sizes, counts


# Example usage 


# Perform box counting
#sizes, counts = box_count(pruned_scenes[7])

# Convert to logarithmic scale
#log_sizes = np.log(sizes)
#log_counts = np.log(counts)

# Plot the results
#plt.figure()
#plt.plot(log_sizes, log_counts, 'o-', label='Data')

# Linear fit
#coefficients = np.polyfit(log_sizes, log_counts, 1)
#fractal_dimension = -coefficients[0]
#poly = np.poly1d(coefficients)
#plt.plot(log_sizes, poly(log_sizes), 'r--', label=f'Fit (D = {fractal_dimension:.4f})')

#plt.xlabel('log(Box Size)')
#plt.ylabel('log(Count)')
#plt.legend()
#plt.title('Box Counting Method')
#plt.show()

#print(f'Estimated Fractal Dimension: {fractal_dimension:.4f}')


def boxCountScenes(scenes, min_box_size=2):
    """
    Apply the boxCount function to a list of binary images (scenes).

    Args:
    scenes (list of numpy.ndarray): List of 2D binary images.
    min_box_size (int): Minimum box size to use in the analysis.

    Returns:
    list of dicts: Each dict contains the results for a scene.
    """
    scene_results = []

    for scene_index, binary_image in enumerate(scenes):
        # Ensure the image is binary
        if binary_image.dtype != bool:
            binary_image = binary_image > 0

        # Apply box counting
        sizes, counts = boxCount(binary_image, min_box_size)

        # Convert to logarithmic scale
        log_sizes = np.log(sizes)
        log_counts = np.log(counts)

        # Perform linear fit to estimate fractal dimension
        coefficients = np.polyfit(log_sizes, log_counts, 1)
        fractal_dimension = -coefficients[0]
        intercept = coefficients[1]

        # Store results in a dictionary
        scene_result = {
            "scene_index": scene_index,
            "fractal_dimension": fractal_dimension,
            "log_sizes": log_sizes,
            "log_counts": log_counts,
            "coefficients": coefficients,
            "sizes": sizes,
            "counts": counts
        }

        scene_results.append(scene_result)

    return scene_results

def batchProcessBoxCount(file_list, folder_path, min_box_size=2):
    """
    Batch processes files, including preprocessing, and compiles results into a pandas DataFrame.

    Args:
    file_list (list): List of file names to be processed.
    folder_path (str): Path to the folder containing the files.
    min_box_size (int): Minimum box size to use in the analysis.

    Returns:
    pd.DataFrame: DataFrame containing results from all processed files.
    """
    # List to store DataFrame rows before concatenation
    all_rows = []

    for file_name in file_list:
        # Construct full file path
        file_path = os.path.join(folder_path, file_name)
        
        # Preprocess the file and get the list of 2D scenes
        scenes = processFileforFrac(file_path)
        
        # Process the scenes to calculate fractal dimensions
        scene_results = boxCountScenes(scenes, min_box_size)
        del scenes  # Free memory

        # Collect each scene's data in all_rows list
        for result in scene_results:
            row = {
                "file_name": file_name,
                "scene_index": result["scene_index"],
                "fractal_dimension": result["fractal_dimension"],
                # Additional data can be included if needed
                # "log_sizes": result["log_sizes"],
                # "log_counts": result["log_counts"],
                # "coefficients": result["coefficients"],
                # "sizes": result["sizes"],
                # "counts": result["counts"]
            }
            all_rows.append(row)
        
        print(f"Finished processing {file_name}")
    
    # Convert list of dicts to DataFrame
    results_df = pd.DataFrame(all_rows)
    
    return results_df


result_frac_df = batchProcessBoxCount(file_list, folder_with_data, min_box_size=2)

result_frac_df




