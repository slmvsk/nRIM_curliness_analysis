#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 08:58:01 2024

@author: tetianasalamovska
"""

####################################################################################
#reading czi files
# pip install czifile 
# to import the package you need to use import czifile
# https://pypi.org/project/czifile/

import czifile

img = czifile.imread('image.czi')
print(img.shape)


import czifile
from skimage import io

img = czifile.imread('images/01.czi')
print(img.shape)
img1=img[0, 0, :, :, :, 0]
print(img1.shape)
img2=img1[2,:,:]
io.imshow(img2)

######################################################################################
### Reading multiple images from a folder
#The glob module finds all the path names 
#matching a specified pattern according to the rules used by the Unix shell
#The glob.glob returns the list of files with their full path 


#import the library opencv
import cv2
import glob

#select the path
path = "images/*.*"
for file in glob.glob(path):
    print(file)     #just stop here to see all file names printed
    a= cv2.imread(file)  #now, we can read each file since we have the full path
    print(a)  #print numpy arrays for each file

#let us look at each file
#    cv2.imshow('Original Image', a)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    

#######################################################################################
### Reading zeiss metadata 

from pylibCZIrw import czi as pyczi
import json

input_image_path = 'image.czi'

# open the CZI for reading using a context manager (preferred way to do it)
# and print the xml metadata
with pyczi.open_czi(input_image_path) as czidoc:
    # get the raw metadata as XML
    md_xml = czidoc.raw_metadata
    print(md_xml)


# Instead of sifting through the extensive metadata, let us print what we need.
# In this case, print the information related to all Channels
with pyczi.open_czi(input_image_path) as czidoc:
    # get the raw metadata as a dictionary
    md_dict = czidoc.metadata

    # Print something specific, like the channel information
    print(json.dumps(md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"], sort_keys=False, indent=4))
# ....
    

# Let us print more metadata
with pyczi.open_czi(input_image_path) as czidoc:
    # get the image dimensions as an dictionary, where the key identifies the dimension
    total_bounding_box = czidoc.total_bounding_box
    print("Image dimensions are: ", total_bounding_box)
    # get the pixel type for all channels
    pixel_types = czidoc.pixel_types
    print("The pixel types for all channels are: ", pixel_types)


# Let us have a look at the image corresponding to the above metadata/ 
#To read a plane from the czi file
with pyczi.open_czi(input_image_path) as czidoc:
    # define some plane coordinates
    plane_0 = {'C': 0, 'Z': 0, 'T': 0}
    plane_1 = {'C': 1, 'Z': 0, 'T': 0}
    plane_2 = {'C': 2, 'Z': 0, 'T': 0}

    channel_0 = czidoc.read(plane=plane_0)
    channel_1 = czidoc.read(plane=plane_1)
    channel_2 = czidoc.read(plane=plane_2)


fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Plot the first image on the first subplot
axes[0,0].imshow(channel_0[:,:,0], cmap='gray')
axes[0,0].axis('off')

# Plot the second image on the second subplot
axes[0,1].imshow(channel_1[:,:,0], cmap='gray')
axes[0,1].axis('off')

# Plot the third image on the third subplot
axes[1,0].imshow(channel_2[:,:,0], cmap='gray')
axes[1,0].axis('off')

axes[1, 1].axis('off')

# Adjust the layout
fig.tight_layout()

# Show the plot
plt.show()


#############################################################################
