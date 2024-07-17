#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 10:53:21 2024

@author: tetianasalamovska
"""

#import matplotlib.pyplot as plt


def plot_slice_from_stack(image_stack, slice_index):
    """
    Plot a specific slice from a given 3D image stack.

    Parameters:
    - image_stack (numpy.ndarray): A 3D numpy array where each slice along the first dimension is an image.
    - slice_index (int): Index of the slice to display.
    """
    # Validate the slice index
    if slice_index < 0 or slice_index >= image_stack.shape[0]:
        raise ValueError(f"Slice index out of range. Provided index: {slice_index}, but should be between 0 and {image_stack.shape[0] - 1}")

    # Plot the specified slice
    plt.figure(figsize=(8, 8))  # Set the figure size
    plt.imshow(image_stack[slice_index, :, :], cmap='gray', aspect='auto')
    plt.title(f'Slice {slice_index + 1}')
    plt.axis('off')  # Hide axes
    plt.show()
    
    
#Example usage assuming 'scenes' and 'adjusted_scenes' are your lists of 3D numpy arrays
#slice_index = 5  # Specify the slice index you want to compare
#if slice_index < len(scenes[1]) and slice_index < len(adjusted_scenes[1]):
    #plot_specific_slice_from_stack(scenes[1], slice_index)
    #plot_specific_slice_from_stack(adjusted_scenes[1], slice_index)
#else:
    #print("Specified slice index is out of range.")
    