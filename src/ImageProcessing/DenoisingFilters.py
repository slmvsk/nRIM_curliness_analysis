#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 10:33:40 2024

@author: tetianasalamovska
"""

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from skimage.filters import gaussian
from skimage import exposure
from skimage.morphology import binary_closing, ball, binary_opening

def applyGaussian(scenes, sigma=1):
    """
    Apply Gaussian blur to all scenes in a list of 3D image stacks.
    
    Parameters:
        scenes (list of numpy.ndarray): List of 3D numpy arrays where each array represents a scene.
        sigma (float or sequence of floats): Standard deviation for Gaussian kernel.
    
    Returns:
        list of numpy.ndarray: List of 3D numpy arrays with Gaussian blur applied.
    """
    blurred_scenes = []
    for i, scene in enumerate(scenes):
        # Apply Gaussian blur to the current scene
        blurred_scene = gaussian_filter(scene, sigma=sigma)
        blurred_scenes.append(blurred_scene)
        print(f"Applied Gaussian blur to scene {i+1}/{len(scenes)}")
    
    return blurred_scenes

# Example usage:
# Assuming 'scenes' is your list of 3D numpy arrays
#blurred_scenes = applyGaussian(scenes, sigma=2)


def applyMedianFilter(scenes, size=3):
    """
    Apply a median filter to all scenes in a list of 3D image stacks.
    
    Parameters:
        scenes (list of numpy.ndarray): List of 3D numpy arrays where each array represents a scene.
        size (int or sequence of ints): Size of the median filter. The filter size should be odd and positive.
    
    Returns:
        list of numpy.ndarray: List of 3D numpy arrays with the median filter applied.
    """
    filtered_scenes = []
    for i, scene in enumerate(scenes):
        # Apply median filter to the current scene
        filtered_scene = median_filter(scene, size=size)
        filtered_scenes.append(filtered_scene)
        print(f"Applied median filter to scene {i+1}/{len(scenes)}")
    
    return filtered_scenes

# Example usage:
#median_scenes = applyMedianFilter(scenes, size=3)

def applyContrastStretching(scenes, lower_percentile=2, upper_percentile=98):
    """
    Apply contrast stretching to all scenes in a list of 3D image stacks.
    
    Parameters:
        scenes (list of numpy.ndarray): List of 3D numpy arrays where each array represents a scene.
        lower_percentile (float): Lower percentile for contrast stretching (default is 2).
        upper_percentile (float): Upper percentile for contrast stretching (default is 98).
    
    Returns:
        list of numpy.ndarray: List of 3D numpy arrays with contrast stretching applied.
    """
    stretched_scenes = []
    for i, scene in enumerate(scenes):
        # Compute the lower and upper percentiles for contrast stretching
        p2, p98 = np.percentile(scene, (lower_percentile, upper_percentile))
        
        # Apply contrast stretching
        stretched_scene = exposure.rescale_intensity(scene, in_range=(p2, p98))
        stretched_scenes.append(stretched_scene)
        
        print(f"Applied contrast stretching to scene {i+1}/{len(scenes)}")
    
    return stretched_scenes


#from skimage.morphology import white_tophat, ball

#selem = ball(radius=12) #try larger radius 
#enhanced_image = white_tophat(subtracted_scenes[7], footprint=selem)
#plotToCompare(subtracted_scenes[7][10,:,:], enhanced_image[10,:,:], 'Substracted', 'Filter')


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



