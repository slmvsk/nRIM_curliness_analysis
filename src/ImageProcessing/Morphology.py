#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 17:43:49 2024

@author: tetianasalamovska
"""
import numpy as np
from scipy.ndimage import binary_erosion


def applyClosing(image, radius=2):
    """
    Apply morphological closing to a 3D image.
    
    Parameters:
        image (numpy.ndarray): The 3D binary image to process.
        radius (int): The radius of the structuring element used for closing.
    
    Returns:
        numpy.ndarray: The 3D image after applying morphological closing.
    """
    # Create a spherical structuring element
    structuring_element = ball(radius)
    
    # Apply morphological closing
    closed_image = binary_closing(image, footprint=structuring_element)
    
    return closed_image



def applyOpening(image, radius=2):
    """
    Apply morphological opening to a 3D image.
    
    Parameters:
        image (numpy.ndarray): The 3D binary image to process.
        radius (int): The radius of the structuring element used for opening.
    
    Returns:
        numpy.ndarray: The 3D image after applying morphological opening.
    """
    # Create a spherical structuring element
    structuring_element = ball(radius)
    
    # Apply morphological opening
    opened_image = binary_opening(image, footprint=structuring_element)
    
    return opened_image



def applyErosion3d(binary_image, iterations=1, structure=None):
    """
    Apply erosion to a 3D binary image.

    Parameters:
        binary_image (ndarray): A 3D binary image.
        iterations (int): Number of iterations to apply erosion.
        structure (ndarray): Structuring element used for erosion (default is a cube).

    Returns:
        eroded_image (ndarray): The eroded 3D binary image.
    """
    # Apply erosion
    eroded_image = binary_erosion(binary_image, structure=structure, iterations=iterations).astype(binary_image.dtype)
    
    return eroded_image

def applyErosionToScenes(scenes, iterations=1, structure=None):
    """
    Apply erosion to each 3D binary image (scene) in a list of scenes.

    Parameters:
        scenes (list): List of 3D binary images.
        iterations (int): Number of iterations to apply erosion.
        structure (ndarray): Structuring element used for erosion (default is a cube).

    Returns:
        processed_scenes (list): List of eroded 3D binary images.
    """
    processed_scenes = []
    
    for i, scene in enumerate(scenes):
        print(f"Processing scene {i+1}/{len(scenes)}")

        # Apply erosion to the current scene
        eroded_scene = applyErosion3d(scene, iterations=iterations, structure=structure)
        processed_scenes.append(eroded_scene)
    
    return processed_scenes


# eroded_scenes = apply_erosion_to_all_scenes(scenes, iterations=2, structure=np.ones((3, 3, 3)))  # Apply erosion with a 3x3x3 structuring element