#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:20:11 2024

@author: tetianasalamovska
"""
pip install aicsimageio
from aicsimageio.readers import CziReader
pip install pylibCZIrw czitools
pip install czitools
from pylibCZIrw import czi as pyczi
from czitools.metadata_tools import czi_metadata as czimd
import numpy as np
from aicsimageio import AICSImage


# import the required libraries
from czitools.metadata_tools import czi_metadata as czimd
from czitools.utils import misc
from ipyfilechooser import FileChooser
from IPython.display import display, HTML
from pathlib import Path
import os
import requests
import ipywidgets as widgets
import glob
import sys
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
        
        # Extract metadata as one big class
        #mdata = czimd.CziMetadata(file_path)
        
        # get the CZI metadata dictionary directly from filename
        #mdict = czimd.create_md_dict_red(mdata, sort=False, remove_none=True)
        # should i save as excel and add widget? 
        
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
        metadata_dict = {
            "Series count": num_scenes,
            "SizeX": czi_dimensions.SizeX,
            "SizeY": czi_dimensions.SizeY,
            "SizeZ": czi_dimensions.SizeZ,
            #"ObjectiveMagnification": objective['NominalMagnification'] if isinstance(objective, dict) else objective[0]['NominalMagnification'],
            #"Zoom": objective['Zoom'] if isinstance(objective, dict) else objective[0]['Zoom'],
            #"ObjectiveNA": objective['LensNA'] if isinstance(objective, dict) else objective[0]['LensNA'],
            #"ExcitationWavelength": illumination_source['Wavelength'] if isinstance(illumination_source, dict) else illumination_source[0]['Wavelength'],
            #"LaserPower": illumination_source['Power'] if isinstance(illumination_source, dict) else illumination_source[0]['Power']
        }

    return scenes, metadata
    
    
        # Extract specific metadata fields
   series_count = first_scene_metadata['SeriesCount']
   size_x = first_scene_metadata['SizeX']
   size_y = first_scene_metadata['SizeY']
   size_z = first_scene_metadata['SizeZ']
   size_s = first_scene_metadata['SizeS']
   scale_x = first_scene_metadata['ScalingX']
   scale_y = first_scene_metadata['ScalingY']
   scale_z = first_scene_metadata['ScalingZ']
   obj_mag = first_scene_metadata['ObjectiveMag']
   zoom = first_scene_metadata['MagnificationZoom']
   obj_na = first_scene_metadata['ObjectiveNA']
   laser_wl = first_scene_metadata['Wavelength']
   em_wl = first_scene_metadata['EmissionWavelength']
   date = first_scene_metadata['CreationDate']
   depth = first_scene_metadata['PixelType']
   positions = all_scenes_metadata['Positions']
   
   # Prepare metadata dictionary
   metadata = {
       'SeriesCount': series_count,
       'SizeX': size_x,
       'SizeY': size_y,
       'SizeZ': size_z,
       'ScaleX': scale_x,
       'ScaleY': scale_y,
       'ScaleZ': scale_z,
       'ObjectiveMag': obj_mag,
       'MagnificationZoom': zoom,
       'ObjectiveNA': obj_na,
       'LaserWavelength': laser_wl,
       'LaserPower': laser_power
   }
   
   return scenes, metadata



file_path = '/Users/tetianasalamovska/Desktop/zeis/IHCT_THT53_40x2x_IHCT08_slice6_stack_positions_A488_laser08_speed6.czi'

# Read CZI file
scenes, metadata = read_czi_stack(file_path)

# Print the metadata
print(metadata)

# Print the shape of the first scene
if scenes:
    print(scenes[0].shape)
