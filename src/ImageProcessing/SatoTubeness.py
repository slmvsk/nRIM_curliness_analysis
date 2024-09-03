#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:28:02 2024

@author: tetianasalamovska
"""


##########################
# Tubeness problem:
# 1. It wants to see image with only tubular structures, so sometimes it connects dots
# 2. Also sigma is dependent on the scale of image so I need to make function "smart" to be
# able to accept different scale as input and adjust sigma?
# Solution: 
# 1. Preprocess better, dots and small onjects doesn't give me any information (dendrite pieces)
# 2. 
##########################

# Here used to be different function that will calculate hessian matrix firstly, but 
# it didn't work properly, so I replaced with automated Sato function 

import numpy as np
from skimage.filters import sato
from skimage import exposure


def applySatoTubeness(image_stack, sigma, black_ridges=True, mode='reflect', cval=0):
    """
    Apply the Sato tubeness filter to a 3D image stack.

    Parameters:
    - image_stack: ndarray (M, N, P)
        3D numpy array representing the image stack.
    - sigmas: iterable of floats, optional
        Sigmas used as scales of the filter. Default is range(1, 10, 2).
    - black_ridges: boolean, optional
        When True, the filter detects black ridges; when False, it detects white ridges. Default is True.
    - mode: {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders. Default is 'reflect'.
    - cval: float, optional
        Used in conjunction with mode 'constant', the value outside the image boundaries. Default is 0.

    Returns:
    - tubeness_image: ndarray (M, N, P)
        The filtered image stack, where the maximum response across all scales is kept.
    """
    # Ensure the input is a 3D numpy array
    if image_stack.ndim != 3:
        raise ValueError("The input image stack must be a 3D numpy array.")

    # If sigma is a single float, convert it to a list
    if isinstance(sigma, (int, float)):
        sigma = [sigma]
    
    # Apply the Sato tubeness filter
    tubeness_image = sato(image_stack, sigmas=sigma, black_ridges=black_ridges, mode=mode, cval=cval)
    
    return tubeness_image

# Validation of tubeness 
def normalizeIntensityZero(image):
    """Normalize the intensity of a 3D image."""
    image_normalized = exposure.rescale_intensity(image, out_range=(0, 1))
    return image_normalized

def subtractTubenessFromImage(image, tubeness):
    """
    Normalize the intensity of a single 3D nosoma image and a 3D tubeness image,
    and subtract the tubeness image from the nosoma image.
    
    Parameters:
        nosoma (ndarray): 3D numpy array with somas removed.
        tubeness (ndarray): 3D numpy array with tubeness measured.
    
    Returns:
        ndarray: A 3D numpy array after subtraction.
    """
    # Normalize both images
    image_normalized = normalizeIntensityZero(image)
    tubeness_normalized = normalizeIntensityZero(tubeness)
    
    # Subtract the tubeness image from the nosoma image
    result = image_normalized - tubeness_normalized # swap to see if there are extra tubeness
    
    # Clip values to keep them in a valid range
    result = np.clip(result, 0, 1)
    
    # Release memory
    del image, tubeness
    gc.collect()
    
    return result

# Example usage:
#validation = subtractTubenessFromImage(nosoma_scenes[8], tubeness_scenes[8])