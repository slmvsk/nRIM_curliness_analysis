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
    straightness = max_dendritic_reach / longest_path_length
    curliness = 1 - straightness

    # Calculate average, std, and sem of curliness
    mean_straightness = np.mean(straightness)
    mean_curliness = np.mean(curliness)
    std_curliness = np.std(curliness)
    median_curliness = np.median(curliness)
    sem_curliness = std_curliness / np.sqrt(len(longest_path_length))

    return curliness, median_curliness, mean_straightness, mean_curliness, sem_curliness, longest_path_length.tolist(), max_dendritic_reach.tolist()


###########
# 3d 
from skimage.measure import label, regionprops
import numpy as np

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
curliness, median_curliness, mean_straightness, mean_curliness, sem_curliness, branch_distances, branch_lengths = analyze_dendrite_curliness_3d(cleaned_skeleton)
# Retrieve the image data for scene index 2
# We assume that 'file_name' or another identifier may be needed if there are multiple entries for the same scene index.
# Here, I'm directly accessing by index if the scene index is used as a row index. If not, you'd filter by conditions.
image_data = dataframe_results[dataframe_results['scene_index'] == 10]['cleaned_scene'].iloc[0]


# Now apply the analyze function
properties, curliness, median_curliness, mean_straightness, mean_curliness, sem_curliness, longest_path_length, max_dendritic_reach = analyze_dendrite_curliness(image_data)
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
curliness_df = analyze_dendrite_curliness_batch(dataframe_results)
print(curliness_df.head())
curliness_df.to_csv('/Users/tetianasalamovska/Desktop/zeis/curliness_df.csv', index=False)

############# COLOR CODE
# label skeletons to see how it identifies dendrites 


import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from skimage import color

def visualize_and_analyze_branches(image, curliness, longest_path_length, max_dendritic_reach):
    # Label the skeleton
    labeled_skeleton = measure.label(image)
    norm = Normalize(vmin=0, vmax=1)  # Normalize the curliness values to [0, 1] for coloring

    # Create a color map
    cmap = plt.cm.viridis  # Use the viridis color map
    
    # Create a color image where each label is colored by its curliness
    colored_skeleton = np.zeros((*image.shape, 3))
    properties = measure.regionprops(labeled_skeleton)
    for prop, curl in zip(properties, curliness):
        color = cmap(norm(curl))[:3]  # Get the RGB values
        colored_skeleton[tuple(prop.coords.T)] = color

    # Plot the colored skeleton
    plt.figure(figsize=(8, 6))
    plt.imshow(colored_skeleton)
    plt.title('Dendrites Colored by Curliness')
    plt.axis('off')
    plt.show()

    # Plot histograms
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.hist(curliness, bins=20, color='purple', alpha=0.7)
    plt.title('Curliness Distribution')
    plt.xlabel('Curliness')
    plt.ylabel('Frequency')

    plt.subplot(2, 2, 2)
    straightness = 1 - np.array(curliness)
    plt.hist(straightness, bins=20, color='blue', alpha=0.7)
    plt.title('Straightness Distribution')
    plt.xlabel('Straightness')

    plt.subplot(2, 2, 3)
    plt.hist(longest_path_length, bins=10000, color='green', alpha=0.7)
    plt.xlim([0, 200])  # Set x-axis limits for example 
    plt.title('Longest path length ')
    plt.xlabel('Length')

    plt.subplot(2, 2, 4)
    plt.hist(max_dendritic_reach, bins=200, color='red', alpha=0.7)
    plt.xlim([0, 300])  # Set x-axis limits for example 
    plt.title('Maximum Dendritic Reach Distribution')
    plt.xlabel('Reach')
    
    plt.tight_layout()
    plt.show()
    
    
# Example assuming you have an 'image_data' which is a binary skeletonized image
visualize_and_analyze_branches(image_data, curliness, longest_path_length, max_dendritic_reach)








############
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable  # Correct import for ScalarMappable
import numpy as np
from skimage import measure, color

def visualize_and_analyze_curliness(image, curliness):
    # Label the skeleton
    labeled_skeleton = measure.label(image)
    norm = Normalize(vmin=0, vmax=1)  # Normalize the curliness values to [0, 1] for coloring
    
    # Use a colormap with high diversity, like 'jet' (alternatively, 'nipy_spectral' can be used)
    cmap = plt.cm.nipy_spectral  # 'jet' or 'nipy_spectral' for high color diversity

    # Apply color map to labeled regions based on curliness
    colored_skeleton = np.zeros((*image.shape, 3))
    properties = measure.regionprops(labeled_skeleton)
    for prop, c in zip(properties, curliness):
        color = cmap(norm(c))[:3]  # Map curliness to RGB color
        colored_skeleton[tuple(prop.coords.T)] = color  # Apply color to the coordinates of each component

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
    n, bins, patches = plt.hist(curliness, bins=30, alpha=0.7)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for bin_center, patch in zip(bin_centers, patches):
        color = cmap(norm(bin_center))
        patch.set_facecolor(color)  # Set the color of each bin to the corresponding curliness value

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
labeled_skeleton = analyze_curliness_3d(cleaned_skeleton, curliness)


#90% of skeleton is recognised as 1 branch, the function is not working correctly then
# I need it to measure branches from node to node (or branch end) instead!!!
# and only then cut outliers 
# I will try graph-based analysis that Yoe mentioned
# which involves representing the skeleton as a graph where junctions 
# are nodes and paths between them are edges !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import numpy as np
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

plot_graph(G)

# Convert skeleton to graph
G = skeleton_to_graph(image_data)


# Plot graph to validate its accuracy
plot_graph(G)


# Print node positions to validate
for node, data in G.nodes(data=True):
    print(f"Node {node} has position {data['pos']}")





import numpy as np
import networkx as netx
from skimage.morphology import skeletonize
from skimage.io import imread


def build_basic_graph(skeleton):
    G = netx.Graph()
    rows, cols = np.where(skeleton > 0)  # Assuming skeleton is a binary image

    # First, add all nodes with their positions
    for y, x in zip(rows, cols):
        G.add_node((y, x), pos=(x, y))  # Map (y, x) to (x, y) for visualization purposes

    # Now add edges based on 8-connectivity
    for y, x in zip(rows, cols):
        # Consider 8 neighboring pixels
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1] and skeleton[ny, nx]:
                if (ny, nx) in G.nodes:
                    # Only add an edge if the neighbor is also a node
                    G.add_edge((y, x), (ny, nx))

    return G

G = build_basic_graph(image_data)





















# Example usage:
# Load your binary image data
# image_data = io.imread('path_to_your_image.png', as_gray=True)
# image_data = img_as_bool(image_data)  # Ensure it is a binary image suitable for skeletonization
results = analyze_dendrite_curliness(image_data)
print(results)

curliness, median_curliness, mean_straightness, mean_curliness, sem_curliness, branch_distances, branch_lengths = analyze_dendrite_curliness(image_data)









import matplotlib.pyplot as plt
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
plot_curliness_results(image_data, results['pixel_curliness'])










########### Box counting method is the best measure (fractality)
# look into curliness (how it identifies branches......)
# try lkshm code with distance matrix 
 











