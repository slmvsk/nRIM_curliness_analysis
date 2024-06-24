#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:20:11 2024

@author: tetianasalamovska
"""

# import the required libraries
# pip install czitools
from czitools.metadata_tools import czi_metadata as czimd
import numpy as np
from czifile import CziFile

def read_czi_stack(file_path):
    """
    Reads a ZEISS CZI file and returns a list of 3D numpy arrays (one per scene) 
    and metadata.
    
    Parameters:
    - file_path (str): Path to the CZI file
    
    Returns:
    - scenes (list of numpy arrays): List of 3D numpy arrays representing image data for each scene (z-stack)
    - metadata (dict): Metadata associated with the CZI file
    """
    with CziFile(file_path) as czi:
        
         # Get the dimensions of the CZI file
        czi_dimensions = czimd.CziDimensions(file_path)
        print("SizeS: ", czi_dimensions.SizeS)
        print("SizeT: ", czi_dimensions.SizeT)
        print("SizeZ: ", czi_dimensions.SizeZ)
        print("SizeC: ", czi_dimensions.SizeC)
        print("SizeY: ", czi_dimensions.SizeY)
        print("SizeX: ", czi_dimensions.SizeX)
        
        num_scenes = czi_dimensions.SizeS if czi_dimensions.SizeS else 1
        #Convert the entire image to a numpy array
        full_image = czi.asarray()
        # Determine the shape of the full image array
        print("Full image shape: ", full_image.shape)
        
        # Initialize list to store 3D numpy arrays for each scene
        scenes = []
        
        # Extract scenes assuming shape (1, 1, S, 1, 1, Z, Y, X, 1)
        for scene_idx in range(num_scenes):
            scene_data = full_image[0, 0, scene_idx, 0, 0, :, :, :, 0]
            scenes.append(scene_data)
            
        # Extract relevant metadata using czitools
        metadata = czimd.CziMetadata(file_path)
        #metadata_dict = {
            #"Series count": num_scenes,
            #"SizeX": czi_dimensions.SizeX,
            #"SizeY": czi_dimensions.SizeY,
            #"SizeZ": czi_dimensions.SizeZ,
            # add more relevant
        #}

    return scenes, metadata

# Read CZI file
#scenes, metadata = read_czi_stack(file_path)
