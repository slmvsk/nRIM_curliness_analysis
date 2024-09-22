#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:16:21 2024

@author: tetianasalamovska
 
"""

# Input image is Z-projection of skeletons 


# Straightness = max. Dendritic reach / longest single path length 
# max. dendritic reach is the shortest path between the start and end points of the branch
# longest single path length is  the actual path length measured along the branch, 
# which will be longer than the branch distance for any curved path
# Curliness will be 1 - straightness that implies that a perfectly straight branch (for ex. 15/15 = 1)
# will have curliness equal 0 (1-1 = 0). More curved branches have higher values approaching 1


import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, io, morphology
from scipy.ndimage import distance_transform_edt
from skimage import measure

# NOTES: break points 
# clean skeleton, keep this method
# Yoe did regionprop for 3D skeletons 
# min branch length as par
# curliness how exactly it looks 
# color by id 
# change lim values for distributions 
# maybe do more strict thresholding if 3d regionprop doestnt work
# if the image before skeletonizing have loops remove


def analyzeCurliness(image):
    # Label the skeleton
    labeled_skeleton = measure.label(image)
    properties = measure.regionprops(labeled_skeleton) # give parameters 

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
    straightness = np.clip(max_dendritic_reach / longest_path_length, 0, 1)
    curliness = 1 - straightness

    # Calculate average, std, and sem of curliness
    mean_straightness = np.mean(straightness)
    mean_curliness = np.mean(curliness)
    std_curliness = np.std(curliness)
    median_curliness = np.median(curliness)
    sem_curliness = std_curliness / np.sqrt(len(longest_path_length))

    return curliness, straightness, longest_path_length.tolist(), max_dendritic_reach.tolist()


# Max dendritic reach (Euclidean distance between endpoints) is greater than the 
# longest path length, which can occur in some edge cases, especially if the skeleton 
# has loops or inaccuracies in labeling the structure.


#!!!!!!!!!!!!! not always length is larger than shortest path


def analyzeCurlinessBatch(scenes_2d):
    """
    Analyze the curliness and other metrics for a list of 2D binary skeletonized images (scenes).

    Args:
    scenes_2d (list): A list of 2D binary skeletonized images.

    Returns:
    list: A list of results for each scene, containing curliness, median curliness, mean straightness,
          mean curliness, SEM of curliness, list of longest path lengths, and list of max dendritic reach.
    """
    all_results = []
    
    for idx, scene in enumerate(scenes_2d):
        print(f"Processing scene {idx+1}/{len(scenes_2d)}")
        
        # Analyze curliness for the current scene
        result = analyzeCurliness(scene)
        
        # Append the result to the list
        all_results.append(result)
    
    return all_results













# 3d 
def analyze_dendrite_curliness_3d(image):
    # Label the 3D skeleton
    labeled_skeleton = label(image)
    properties = regionprops(labeled_skeleton)

    longest_path_length = []
    max_dendritic_reach = []

    # Calculate measures for each region
    for prop in properties:
        # Use volume as a proxy for path length in 3D
        longest_path_length.append(prop.area)  # Replace with prop.volume for true 3D measure
        
        # Calculate straight-line (Euclidean) distance between the end points of the 3D bounding box
        minr, minc, mins, maxr, maxc, maxs = prop.bbox  # mins and maxs are the min and max along the z-axis
        distance = np.sqrt((maxr - minr) ** 2 + (maxc - minc) ** 2 + (maxs - mins) ** 2)
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
# Assuming 'image_3d' is your loaded 3D skeletonized image
# properties, curl






# Example usage
#curliness, median_curliness, mean_straightness, mean_curliness, sem_curliness, branch_distances, branch_lengths = analyze_dendrite_curliness_3d(cleaned_skeleton)
# Retrieve the image data for scene index 2
# We assume that 'file_name' or another identifier may be needed if there are multiple entries for the same scene index.
# Here, I'm directly accessing by index if the scene index is used as a row index. If not, you'd filter by conditions.
#image_data = dataframe_results[dataframe_results['scene_index'] == 10]['cleaned_scene'].iloc[0]


# Now apply the analyze function
#properties, curliness, median_curliness, mean_straightness, mean_curliness, sem_curliness, longest_path_length, max_dendritic_reach = analyze_dendrite_curliness(image_data)
#print("Mean Straightness:", mean_straightness)
#print("Mean Curliness:", median_curliness)
#print("Longest Path Length:", longest_path_length)
#print("Maximum Dendritic Reach:", max_dendritic_reach)



############################# dataframe is an input 

import pandas as pd

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
        # median
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
        # median 
    })

    return results_df

# Assuming 'dataframe_results' is your DataFrame that contains all the processed images.
#curliness_df = analyze_dendrite_curliness_batch(dataframe_results)
#print(curliness_df.head())
#curliness_df.to_csv('/Users/tetianasalamovska/Desktop/zeis/curliness_df.csv', index=False)

############# COLOR CODE
# label skeletons to see how it identifies dendrites 


import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
from skimage import measure

def visualize_and_analyze_branches(image, curliness, longest_path_length, max_dendritic_reach):
    # Label the skeleton
    labeled_skeleton = measure.label(image)
    norm = Normalize(vmin=0, vmax=1)  # Normalize the curliness values to [0, 1] for coloring

    # Create a vibrant and diverse color map
    cmap = plt.cm.nipy_spectral  # Use 'nipy_spectral' for bright, diverse colors
    
    # Create a color image where each label is colored by its curliness
    colored_skeleton = np.zeros((*image.shape, 3))
    properties = measure.regionprops(labeled_skeleton)
    
    for prop, curl in zip(properties, curliness):
        color = cmap(norm(curl))[:3]  # Get the RGB values from the colormap
        # Apply the color to the corresponding segment (branch) in the image
        colored_skeleton[tuple(prop.coords.T)] = color

    # Plot the colored skeleton
    plt.figure(figsize=(8, 6))
    plt.imshow(colored_skeleton)
    plt.title('Dendrites Colored by Curliness')
    plt.axis('off')
    plt.show()

    # Prepare the mappable object for colorbar and histogram
    mappable = ScalarMappable(norm=norm, cmap=cmap)

    # Plot histograms with color coding that corresponds to the curliness in the image
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    colors = mappable.to_rgba(curliness)
    n, bins, patches = plt.hist(curliness, bins=20, alpha=0.7)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    # Color each bin in the histogram based on the curliness
    for bin_center, patch in zip(bin_centers, patches):
        color = cmap(norm(bin_center))
        patch.set_facecolor(color)
    plt.title('Curliness Distribution (Color Matched)')
    plt.xlabel('Curliness')
    plt.ylabel('Frequency')

    plt.subplot(2, 2, 2)
    straightness = 1 - np.array(curliness)
    n, bins, patches = plt.hist(straightness, bins=20, alpha=0.7)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for bin_center, patch in zip(bin_centers, patches):
        color = cmap(norm(1 - bin_center))  # Inverse color matching for straightness
        patch.set_facecolor(color)
    plt.title('Straightness Distribution (Inverse Color Matched)')
    plt.xlabel('Straightness')

    plt.subplot(2, 2, 3)
    plt.hist(longest_path_length, bins=100, color='green', alpha=0.7)
    plt.xlim([0, 200])  # Set x-axis limits for example 
    plt.title('Longest Path Length Distribution')
    plt.xlabel('Length')

    plt.subplot(2, 2, 4)
    plt.hist(max_dendritic_reach, bins=100, color='red', alpha=0.7)
    plt.xlim([0, 300])  # Set x-axis limits for example 
    plt.title('Maximum Dendritic Reach Distribution')
    plt.xlabel('Reach')

    plt.tight_layout()
    plt.show()

    # Adding a colorbar to show the color coding for curliness
    plt.figure(figsize=(6, 1))
    plt.colorbar(mappable, orientation='horizontal', label='Curliness (0 to 1)')
    plt.show()







############
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable  # Correct import for ScalarMappable

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from skimage import measure
import numpy as np

def visualize_and_analyze_curliness(image, curliness):
    """
    Visualize skeleton segments color-coded by their curliness and plot a curliness histogram.
    
    Parameters:
        image (numpy.ndarray): 2D binary skeletonized image.
        curliness (list): List of curliness values for each segment.
    """
    
    # Label the skeleton
    labeled_skeleton = measure.label(image)
    norm = Normalize(vmin=0, vmax=1)  # Normalize the curliness values to [0, 1] for coloring
    
    # Use a colormap with high diversity, like 'jet' or 'nipy_spectral'
    cmap = plt.cm.nipy_spectral  # You can switch to 'jet' or any other colormap

    # Apply color map to labeled regions based on curliness
    colored_skeleton = np.zeros((*image.shape, 3))  # To store RGB values for the skeleton
    properties = measure.regionprops(labeled_skeleton)
    
    for prop, c in zip(properties, curliness):
        color = cmap(norm(c))[:3]  # Map curliness to RGB color
        colored_skeleton[tuple(prop.coords.T)] = color  # Apply color to the coordinates of each segment

    # Display the colorized skeleton
    plt.figure(figsize=(8, 6))
    plt.imshow(colored_skeleton)
    plt.title('Dendrites Colored by Curliness')
    plt.axis('off')
    plt.show()

    # Plotting curliness histogram with the same color coding
    plt.figure(figsize=(8, 6))
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    colors = mappable.to_rgba(curliness)
    
    # Create a histogram with color coding
    n, bins, patches = plt.hist(curliness, bins=30, alpha=0.7, edgecolor='black')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for bin_center, patch in zip(bin_centers, patches):
        color = cmap(norm(bin_center))  # Map bin center to color
        patch.set_facecolor(color)  # Set bin color based on curliness

    # Add a color bar for the histogram
    plt.colorbar(mappable, label='Curliness')
    plt.title('Histogram of Curliness Distribution')
    plt.xlabel('Curliness')
    plt.ylabel('Frequency')
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def analyze_curliness_3d(image, curliness):
    """
    Analyze curliness in a 3D skeletonized image, generate a histogram of curliness values,
    and return the labeled skeleton.

    Parameters:
        image (numpy.ndarray): The 3D binary skeletonized image.
        curliness (list): A list of curliness values for each labeled region.

    Returns:
        numpy.ndarray: The labeled skeleton where each connected component has a unique label.
    """
    # Label the skeleton in 3D
    labeled_skeleton = measure.label(image, connectivity=3)
    norm = Normalize(vmin=0, vmax=1)  # Normalize curliness values to [0, 1] for coloring
    
    # Use a colormap with high diversity, like 'nipy_spectral'
    cmap = plt.cm.nipy_spectral

    # Create a figure and axis for the histogram
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plotting curliness histogram with color coding
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    colors = mappable.to_rgba(curliness)
    
    # Create a histogram with color coding
    n, bins, patches = ax.hist(curliness, bins=30, alpha=0.7)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for bin_center, patch in zip(bin_centers, patches):
        color = cmap(norm(bin_center))
        patch.set_facecolor(color)  # Set the color of each bin to the corresponding curliness value

    # Add colorbar to the axis
    fig.colorbar(mappable, ax=ax, label='Curliness')

    ax.set_title('Histogram of Curliness Distribution')
    ax.set_xlabel('Curliness')
    ax.set_ylabel('Frequency')
    
    plt.show()

    return labeled_skeleton



# Example usage:
#labeled_skeleton = analyze_curliness_3d(cleaned_skeleton, curliness)


#90% of skeleton is recognised as 1 branch, the function is not working correctly then
# I need it to measure branches from node to node (or branch end) instead!!!
# and only then cut outliers 
# I will try graph-based analysis that Yoe mentioned
# which involves representing the skeleton as a graph where junctions 
# are nodes and paths between them are edges !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


import networkx as netx

def skeleton_to_graph(skeleton):
    G = netx.Graph()
    
    # Iterate through each point in the skeleton
    rows, cols = np.where(skeleton)  # Find all skeleton points
    for y, x in zip(rows, cols):
        # Create a unique identifier for the node using its coordinates
        node_id = (y, x)
        if node_id not in G:
            G.add_node(node_id, pos=(x, y))  # Store positions in 'pos' attribute using Cartesian (x, y) format
        
        # Check for direct neighbors (8-connectivity)
        neighbors = [(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
        for ny, nx in neighbors:
            neighbor_id = (ny, nx)
            if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1] and skeleton[ny, nx]:
                if neighbor_id not in G:
                    G.add_node(neighbor_id, pos=(nx, ny))
                if not G.has_edge(node_id, neighbor_id):
                    G.add_edge(node_id, neighbor_id)

    return G



def plot_graph(G):
    pos = {node: (node[1], -node[0]) for node in G.nodes()}  # Flip y-axis for display
    nx.draw(G, pos, node_size=0.01, edge_color='black', node_color='red', with_labels=False)
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.show()

#plot_graph(G)

# Convert skeleton to graph
#G = skeleton_to_graph(image_data)


# Plot graph to validate its accuracy
#plot_graph(G)


# Print node positions to validate
#for node, data in G.nodes(data=True):
    #print(f"Node {node} has position {data['pos']}")


















from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from skimage.color import gray2rgb

def plot_curliness_results(skeleton, curliness):
    norm = Normalize(vmin=curliness.min(), vmax=curliness.max())
    cmap = plt.cm.plasma  # High variation color map

    # Color the skeleton
    skeleton_rgb = gray2rgb(skeleton)
    colored_skeleton = np.zeros_like(skeleton_rgb)
    indices = np.argwhere(skeleton)
    colors = cmap(norm(curliness[indices[:, 0], indices[:, 1]]))
    for i in range(3):  # Apply RGB channels
        colored_skeleton[indices[:, 0], indices[:, 1], i] = colors[:, i]

    plt.figure(figsize=(10, 8))
    plt.imshow(colored_skeleton)
    plt.title('Dendrites Colored by Curliness')
    plt.axis('off')
    plt.show()

    # Histogram of curliness
    plt.figure(figsize=(8, 6))
    plt.hist(curliness.ravel(), bins=50, color='blue', alpha=0.7)
    plt.title('Histogram of Curliness Distribution')
    plt.xlabel('Curliness')
    plt.ylabel('Frequency')
    plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), label='Curliness')
    plt.show()
# Assuming `image_data` and `results['curliness']` are prepared
#plot_curliness_results(image_data, results['pixel_curliness'])












