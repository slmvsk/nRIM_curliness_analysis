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


from skimage import measure, io, morphology
from scipy.ndimage import distance_transform_edt

# NOTES: break points 
# clean skeleton, keep this method
# Yoe did regionprop for 3D skeletons 
# min branch length as par
# curliness how exactly it looks 
# color by id 
# change lim values for distributions 
# maybe do more strict thresholding if 3d regionprop doestnt work
# if the image before skeletonizing have loops remove


import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import networkx as nx
from skimage import measure, morphology
from scipy.spatial.distance import euclidean

def compute_geodesic_length(branch_mask, start, end):
    import networkx as nx

    # Build the graph from the skeleton
    G = nx.Graph()
    indices = np.transpose(np.nonzero(branch_mask))
    for idx in indices:
        y, x = idx
        # Consider all 8-connected neighbors
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < branch_mask.shape[0] and 0 <= nx_ < branch_mask.shape[1]:
                    if branch_mask[ny, nx_]:
                        # Euclidean distance between pixels (1 for orthogonal, sqrt(2) for diagonal)
                        weight = np.hypot(dy, dx)
                        G.add_edge((y, x), (ny, nx_), weight=weight)
    # Compute shortest path length
    try:
        geodesic_length = nx.shortest_path_length(G, source=start, target=end, weight='weight')
    except nx.NetworkXNoPath:
        geodesic_length = 0
    return geodesic_length


def analyzeCurliness(image):
    import numpy as np
    from skimage import measure
    from scipy.spatial.distance import euclidean
    import networkx as nx

    # Label the skeleton
    labeled_skeleton = measure.label(image)
    properties = measure.regionprops(labeled_skeleton)

    longest_path_length = []
    max_dendritic_reach = []
    labels = []

    for prop in properties:
        label = prop.label
        coords = prop.coords
        num_pixels = coords.shape[0]
        if num_pixels < 2:
            continue

        # Create a mask for the current branch
        branch_mask = (labeled_skeleton == label)

        # Find endpoints of the branch
        endpoints = find_endpoints(branch_mask)
        if len(endpoints) < 2:
            continue  # Cannot compute if less than 2 endpoints

        # For branches with multiple endpoints, consider all pairs
        max_geodesic_length = 0
        max_euclidean_dist = 0
        for i in range(len(endpoints)):
            for j in range(i+1, len(endpoints)):
                ep1 = endpoints[i]
                ep2 = endpoints[j]
                # Compute geodesic path length between endpoints
                geodesic_length = compute_geodesic_length(branch_mask, ep1, ep2)
                # Euclidean distance between the two endpoints
                euclidean_dist = euclidean(ep1, ep2)
                if geodesic_length > max_geodesic_length:
                    max_geodesic_length = geodesic_length
                    max_euclidean_dist = euclidean_dist

        if max_geodesic_length == 0:
            continue

        # Append values
        labels.append(label)
        longest_path_length.append(max_geodesic_length)
        max_dendritic_reach.append(max_euclidean_dist)

    longest_path_length = np.array(longest_path_length)
    max_dendritic_reach = np.array(max_dendritic_reach)

    # Compute straightness and curliness
    straightness = np.clip(max_dendritic_reach / longest_path_length, 0, 1)
    curliness = 1 - straightness

    return curliness, straightness, longest_path_length.tolist(), max_dendritic_reach.tolist(), labeled_skeleton, labels

def find_endpoints(skeleton):
    from scipy.ndimage import convolve

    # Ensure the skeleton is binary
    skeleton = skeleton.astype(np.uint8)

    # Define a kernel to count neighbors
    kernel = np.array([[1,1,1],
                       [1,10,1],
                       [1,1,1]])

    # Convolve the skeleton with the kernel
    convolved = convolve(skeleton, kernel, mode='constant', cval=0)

    # For each pixel, subtract 10 (the pixel's own value multiplied by 10 in the kernel center)
    neighbor_count = (convolved - 10) * skeleton

    # Endpoints are pixels with only one neighbor
    endpoints = (neighbor_count == 1)

    # Get the coordinates of endpoints
    endpoint_coords = np.column_stack(np.nonzero(endpoints))

    return [tuple(coord) for coord in endpoint_coords]


#def analyzeCurliness(image):
    # Label the skeleton
    #labeled_skeleton = measure.label(image)
    #properties = measure.regionprops(labeled_skeleton)

    #longest_path_length = [] #geodesic
    #max_dendritic_reach = [] #euclidean
    #labels = []

    #for prop in properties:
        #label = prop.label
        #coords = prop.coords  # N x 2 array of (row, col) coordinates
        #num_pixels = coords.shape[0]
        #if num_pixels < 2:
            #continue  # Skip branches with less than 2 pixels
        # Compute all pairwise Euclidean distances
        #from scipy.spatial.distance import pdist
        #distances = pdist(coords, 'euclidean')
        #max_distance = distances.max()
        # Append values
        #labels.append(label)
        #longest_path_length.append(num_pixels)
        #max_dendritic_reach.append(max_distance)

    #longest_path_length = np.array(longest_path_length)
    #max_dendritic_reach = np.array(max_dendritic_reach)

    # Compute straightness and curliness
    #straightness = np.clip(max_dendritic_reach / longest_path_length, 0, 1)
    #curliness = 1 - straightness

    #return curliness, straightness, longest_path_length.tolist(), max_dendritic_reach.tolist(), labeled_skeleton, labels




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
#def analyze_dendrite_curliness_3d(image):
    # Label the 3D skeleton
    #labeled_skeleton = label(image)
    #properties = regionprops(labeled_skeleton)

    #longest_path_length = []
    #max_dendritic_reach = []

    # Calculate measures for each region
    #for prop in properties:
        # Use volume as a proxy for path length in 3D
        #longest_path_length.append(prop.area)  # Replace with prop.volume for true 3D measure
        
        # Calculate straight-line (Euclidean) distance between the end points of the 3D bounding box
        #minr, minc, mins, maxr, maxc, maxs = prop.bbox  # mins and maxs are the min and max along the z-axis
        #distance = np.sqrt((maxr - minr) ** 2 + (maxc - minc) ** 2 + (maxs - mins) ** 2)
        #max_dendritic_reach.append(distance)

    # Output each branch's measures
    #for length, reach in zip(longest_path_length, max_dendritic_reach):
        #print(f"Branch: Length = {length}, Max Reach = {reach}")

    # Convert lists to arrays for numerical operations
    #longest_path_length = np.array(longest_path_length)
    #max_dendritic_reach = np.array(max_dendritic_reach)

    # Calculate straightness and curliness
    #straightness = max_dendritic_reach / longest_path_length
    #curliness = 1 - straightness

    # Calculate average, std, and sem of curliness
    #mean_straightness = np.mean(straightness)
    #mean_curliness = np.mean(curliness)
    #std_curliness = np.std(curliness)
    #median_curliness = np.median(curliness)
    #sem_curliness = std_curliness / np.sqrt(len(longest_path_length))

    #return curliness, median_curliness, mean_straightness, mean_curliness, sem_curliness, longest_path_length.tolist(), max_dendritic_reach.tolist()



# Here, I'm directly accessing by index if the scene index is used as a row index. If not, you'd filter by conditions.
#image_data = dataframe_results[dataframe_results['scene_index'] == 10]['cleaned_scene'].iloc[0]




############################# dataframe is an input 
# I might not want mean? or normalize it? 


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

def visualize_and_analyze_branches(labeled_skeleton, curliness, labels, longest_path_length, max_dendritic_reach):
    norm = Normalize(vmin=0, vmax=1)
    cmap = plt.cm.nipy_spectral

    # Create a color image where each label is colored by its curliness
    colored_skeleton = np.zeros((*labeled_skeleton.shape, 3))
    for label, curl in zip(labels, curliness):
        color = cmap(norm(curl))[:3]
        colored_skeleton[labeled_skeleton == label] = color

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
    plt.hist(longest_path_length, bins=50, color='green', alpha=0.7)
    plt.title('Longest Path Length Distribution')
    plt.xlabel('Length')

    plt.subplot(2, 2, 4)
    plt.hist(max_dendritic_reach, bins=50, color='red', alpha=0.7)
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
    cmap = plt.cm.jet  # You can switch to 'jet' or any other colormap

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










