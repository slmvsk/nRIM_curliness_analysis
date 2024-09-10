#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 16:15:36 2024

@author: tetianasalamovska
"""

import cv2
import numpy as np

def otsuThresholding(image_stack):
    """
    Apply Otsu's thresholding to a 3D image stack.
    
    Parameters:
        image_stack (numpy.ndarray): A 3D numpy array representing the image stack.
    
    Returns:
        numpy.ndarray: A 3D numpy array of the binarized image stack.
    """
    binarized_stack = np.zeros_like(image_stack, dtype=np.uint8)
    
    for i in range(image_stack.shape[0]):
        _, binarized_stack[i, :, :] = cv2.threshold(
            src=image_stack[i, :, :],
            thresh=0,
            maxval=255,
            type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    
    return binarized_stack


def otsuThresholdingScenes(scenes):
    """
    Apply Otsu's thresholding to each 3D numpy array in a list.
    
    Parameters:
        scenes (list): List of 3D numpy arrays where each array represents a scene.
    
    Returns:
        list: A list of 3D numpy arrays with Otsu's thresholding applied.
    """
    processed_scenes = []
    
    for i, scene in enumerate(scenes):
        print(f"Processing scene {i+1}/{len(scenes)}")
        
        if scene.size == 0:
            print(f"Scene {i+1} is empty or invalid!")
            continue
        
        try:
            processed_scene = otsuThresholding(scene)
            processed_scenes.append(processed_scene)
        except Exception as e:
            print(f"Error processing scene {i+1}: {e}")
    
    print(f"Total processed scenes: {len(processed_scenes)}")
    return processed_scenes
