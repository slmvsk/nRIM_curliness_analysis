#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 16:21:02 2024

@author: tetianasalamovska
"""

def verify_and_plot_intensity(image_stack, adjusted_stack, slice_index):
    # Extract a single slice for analysis
    original_slice = image_stack[:, :, slice_index]
    adjusted_slice = adjusted_stack[:, :, slice_index]

    # Print the min and max values
    print("Original slice - Min:", np.min(original_slice), "Max:", np.max(original_slice))
    print("Adjusted slice - Min:", np.min(adjusted_slice), "Max:", np.max(adjusted_slice))

    # Create histograms of the pixel values
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(original_slice.ravel(), bins=256, color='blue', alpha=0.7)
    axes[0].set_title('Histogram of Original Slice')
    axes[0].set_xlabel('Pixel intensity')
    axes[0].set_ylabel('Count')

    axes[1].hist(adjusted_slice.ravel(), bins=256, color='green', alpha=0.7)
    axes[1].set_title('Histogram of Adjusted Slice')
    axes[1].set_xlabel('Pixel intensity')
    axes[1].set_ylabel('Count')

    plt.show()

# Example usage
#adjusted_first_scene = normalize_intensity(first_scene)
#verify_and_plot_intensity(first_scene, adjusted_first_scene, 5)  # Using the 6th slice for example

