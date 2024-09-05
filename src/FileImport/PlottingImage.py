#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 10:53:21 2024

@author: tetianasalamovska
"""

import matplotlib.pyplot as plt
import numpy as np
from mayavi import mlab

   
def plotComparison(image1, image2, title):
    """
    Plots two images side-by-side for comparison.

    Parameters:
        image1 (ndarray): First image to plot.
        image2 (ndarray): Second image to plot.
        title (str): Title for the subplot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax1, ax2 = axes.ravel()

    ax1.imshow(image1, cmap='gray')
    ax1.set_title('Original')
    ax1.axis('off')  # Turn off axis numbering

    ax2.imshow(image2, cmap='gray')
    ax2.set_title('Filtered')
    ax2.axis('off')  # Turn off axis numbering

    plt.suptitle(title)
    plt.show()

# Example usage:
# Assuming adjusted_scenes[2][5,:,:] and filtered_image[5,:,:] are correctly shaped (1024, 1024)
#plot_comparison(adjusted_scenes[2][5,:,:], filtered_image[5,:,:], "Background Correction Comparison")

# this is the best one so far 
def plotToCompare(image1, image2, title1='Image 1', title2='Image 2'):
    """Plot two images side by side with high quality."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=300)  # Increase dpi for higher quality
    
    # Plot the first image
    ax[0].imshow(image1, cmap='gray')
    ax[0].set_title(title1)
    ax[0].axis('off')  # Hide the axes

    # Plot the second image
    ax[1].imshow(image2, cmap='gray')
    ax[1].set_title(title2)
    ax[1].axis('off')  # Hide the axes

    plt.tight_layout() # Adjust layout to fit images
    plt.show()

#borrowed function to show all 3 projections
def show(image_to_show, labels=False):
    """
    This function generates three projections: in X-, Y- and Z-direction and shows them.
    """
    projection_x = cle.maximum_x_projection(image_to_show)
    projection_y = cle.maximum_y_projection(image_to_show)
    projection_z = cle.maximum_z_projection(image_to_show)

    fig, axs = plt.subplots(1, 3, figsize=(15, 15))
    cle.imshow(projection_x, plot=axs[0], labels=labels)
    cle.imshow(projection_y, plot=axs[1], labels=labels)
    cle.imshow(projection_z, plot=axs[2], labels=labels)

#show(input_gpu)
#print(input_gpu.shape)

def plotImageHistogram(image_stack, bins=256, pixel_range=(0, 65535), title='Pixel Intensity Histogram'):
    """
    Plot the histogram of pixel intensities for a 3D image stack.
    
    Parameters:
        image_stack (numpy.ndarray): A 3D numpy array (Z, Y, X) of an image stack.
        bins (int): Number of bins for the histogram.
        pixel_range (tuple): The range (min, max) of pixel intensities.
        title (str): Title of the histogram plot.
    """
    # Flatten the 3D image stack to a 1D array
    pixels = image_stack.ravel()

    # Calculate the histogram
    histogram, bin_edges = np.histogram(pixels, bins=bins, range=pixel_range)

    # Configure the plot
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Pixel Count')

    # Plot the histogram
    plt.bar(bin_edges[:-1], histogram, width=np.diff(bin_edges), edgecolor='black', align='edge')

    # Display the plot
    plt.show()
    
#plotImageHistogram(scenes[6], bins=256, pixel_range=(0, 65535), title='Pixel Intensity Histogram for Original Image')

def visualize3dMayavi(image):
    """
    Visualize a 3D image using Mayavi.

    Parameters:
        image (numpy.ndarray): The 3D image data.
    """
    if image.dtype == bool:
        image = image.astype(np.int8)
    mlab.contour3d(image, contours=10, opacity=0.5)
    mlab.show()

# Example usage
#visualize_3d_mayavi(skeletonized)
