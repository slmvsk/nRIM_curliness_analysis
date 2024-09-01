#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 10:53:21 2024

@author: tetianasalamovska
"""

import matplotlib.pyplot as plt
   
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