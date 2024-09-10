#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 16:15:36 2024

@author: tetianasalamovska
"""

import numpy as np
import cv2

def adaptiveGaussianThresholding(image_stack):
    """
    Apply adaptive Gaussian thresholding to a 3D image stack.
    
    Parameters:
        image_stack (numpy.ndarray): A 3D numpy array representing the image stack.
    
    Returns:
        numpy.ndarray: A 3D numpy array of the binarized image stack.
    """
    # Initialize a binarized stack
    binarized_stack = np.zeros_like(image_stack, dtype=np.uint8)
    
    # Process each slice using adaptive Gaussian thresholding
    for i in range(image_stack.shape[0]):
        # Apply adaptive thresholding
        binarized_stack[i, :, :] = cv2.adaptiveThreshold(
            src=cv2.convertScaleAbs(image_stack[i, :, :]),  # Convert to 8-bit if not already
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=11,  # Size of a pixel neighborhood used to calculate the threshold value
            C=2  # Constant subtracted from the mean or weighted mean
        )
    
    return binarized_stack

def thresholdScenes(scenes):
    """
    Apply adaptive Gaussian thresholding to each scene in a list.
    
    Parameters:
        scenes (list of ndarray): List of 3D numpy arrays.
    
    Returns:
        list of ndarray: List of binarized 3D numpy arrays.
    """
    processed_scenes = []
    for i, scene in enumerate(scenes):
        print(f"Processing scene {i+1}/{len(scenes)}")
        try:
            processed_scene = adaptive_gaussian_thresholding(scene)
            processed_scenes.append(processed_scene)
            print(f"Processed scene {i+1} successfully.")
        except Exception as e:
            print(f"Error processing scene {i+1}: {e}")
    return processed_scenes

# Example usage:
# Assuming 'scenes' is your list of 3D numpy arrays
processed_scenes = process_all_scenes(scenes)