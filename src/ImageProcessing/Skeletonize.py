#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 07:27:03 2024

@author: tetianasalamovska
"""

import numpy as np
from skimage.morphology import skeletonize_3d
import gc
from skimage import img_as_ubyte
from scipy.spatial.distance import euclidean
from skimage.measure import label, regionprops
import os
import cv2
from plantcv.plantcv import params
from plantcv.plantcv import image_subtract
from plantcv.plantcv.morphology import segment_sort
from plantcv.plantcv.morphology import segment_skeleton
from plantcv.plantcv.morphology import _iterative_prune
from plantcv.plantcv._debug import _debug
from plantcv.plantcv._helpers import _cv2_findcontours


def skeletonizeImage(image):
    """
    Apply skeletonization to a 3D binary image.
    
    Args:
    image (numpy.ndarray): A 3D binary image where the objects are 1's and the background is 0's.
    
    Returns:
    numpy.ndarray: A 3D binary image containing the skeleton of the original image.
    """
    if not np.all(np.isin(image, [0, 1])):
        image = (image > 0).astype(np.uint8)
        print("Warning: Input image was not binary. It has been converted to binary.")
    
    # Apply skeletonization
    skeleton = skeletonize_3d(image)
    return skeleton


def skeletonizeScenes(scenes):
    """
    Process a list of 3D scenes: binarize, skeletonize, and release memory after processing each scene.
    
    Args:
    scenes (list of numpy.ndarray): A list of 3D images where each image is a scene.
    
    Returns:
    list of numpy.ndarray: A list of 3D binary skeletonized images.
    """
    processed_scenes = []
    
    for i, scene in enumerate(scenes):
        print(f"Processing scene {i+1}/{len(scenes)}")
        
        # Binarize and skeletonize the scene
        try:
            binarized_scene = img_as_ubyte(scene > 0)
            skeleton = skeletonizeImage(binarized_scene)
            processed_scenes.append(skeleton)
            
            # Release memory
            del scene
            gc.collect()
        
        except Exception as e:
            print(f"Error processing scene {i+1}: {e}")
            continue
    
    return processed_scenes



#############################################
# z-projection to visualize or analyze in 2D 

def z_projection(image_3d):
    """
    Create a maximum intensity projection (MIP) of a 3D binary image along the z-axis.
    
    Args:
    image_3d (numpy.ndarray): A 3D numpy array representing the binary image. Expected shape is (z, y, x).
    
    Returns:
    numpy.ndarray: A 2D numpy array representing the maximum intensity projection along the z-axis.
    """
    if not np.all(np.isin(image_3d, [0, 1])):
        image_3d = (image_3d > 0).astype(np.uint8)
        print("Warning: Input image was not binary. It has been converted to binary.")
        
    # Compute the maximum intensity projection by using np.max along the z-axis
    mip = np.max(image_3d, axis=0)  # Axis 0 corresponds to the z-direction in your image stack
    return mip



def zProjectScenes(skeletonized_scenes):
    """
    Apply maximum intensity Z projection to each scene in a list of skeletonized 3D images.

    Parameters:
        skeletonized_scenes (list): List of 3D binary numpy arrays representing the skeletonized scenes.

    Returns:
        list: A list of 2D numpy arrays representing the maximum intensity projection for each scene.
    """
    mip_scenes = []
    
    for i, scene in enumerate(skeletonized_scenes):
        print(f"Processing scene {i+1}/{len(skeletonized_scenes)}")
        
        try:
            # Apply max intensity Z projection to the current scene
            mip = z_projection(scene)
            mip_scenes.append(mip)
            
        except Exception as e:
            print(f"Error processing scene {i+1}: {e}")
            continue
        
        # Release memory for the current scene
        del scene
        gc.collect()

    return mip_scenes


##################################################
# cleaning, batch for MIP images (2D)



def cleanMipSkeleton(scenes_2d, min_length=10, max_length=100):
    """
    Clean a list of 2D binary skeleton images by removing branches shorter than `min_length` 
    or longer than `max_length`.
    
    Args:
        scenes_2d (list): A list of 2D binary images of skeletonized structures.
        min_length (float): Minimum branch length to keep.
        max_length (float): Maximum branch length to keep.

    Returns:
        list: A list of cleaned skeleton images with filtered branches.
    """
    cleaned_scenes = []
    
    for idx, image_2d in enumerate(scenes_2d):
        # Ensure the image is binary
        binary_image = (image_2d > 0).astype(np.uint8)
        
        # Label the connected components in the skeleton
        labeled_skeleton = label(binary_image)
        
        # Measure the properties of the labeled regions
        props = regionprops(labeled_skeleton)
        
        # Create a new cleaned skeleton image
        cleaned_skeleton = np.zeros_like(binary_image)
        
        for prop in props:
            coords = prop.coords
            branch_length = 0
            for i in range(len(coords) - 1):
                branch_length += euclidean(coords[i], coords[i + 1])
            
            # Keep branches that meet the specified length criteria
            if min_length <= branch_length <= max_length:
                for coord in coords:
                    cleaned_skeleton[coord[0], coord[1]] = 1  # Set pixel to 1 to keep the branch
        
        cleaned_scenes.append(cleaned_skeleton)
        print(f"Processed scene {idx+1}/{len(scenes_2d)}: Skeleton cleaned, {len(props)} branches processed")
    
    return cleaned_scenes



#######################################
# Prune 2D


def prune2D(skel_img, size=0, mask=None):
    """Prune the ends of skeletonized segments.
    The pruning algorithm proposed by https://github.com/karnoldbio
    Segments a skeleton into discrete pieces, prunes off all segments less than or
    equal to user specified size. Returns the remaining objects as a list and the
    pruned skeleton.

    Inputs:
    skel_img    = Skeletonized image
    size        = Size to get pruned off each branch
    mask        = (Optional) binary mask for debugging. If provided, debug image will be overlaid on the mask.

    Returns:
    pruned_img      = Pruned image
    segmented_img   = Segmented debugging image
    segment_objects = List of contours

    :param skel_img: numpy.ndarray
    :param size: int
    :param mask: numpy.ndarray
    :return pruned_img: numpy.ndarray
    :return segmented_img: numpy.ndarray
    :return segment_objects: list
    """
    # Store debug
    debug = params.debug
    params.debug = None

    pruned_img = skel_img.copy()

    _, objects = segment_skeleton(skel_img)
    kept_segments = []
    removed_segments = []

    if size > 0:
        # If size>0 then check for segments that are smaller than size pixels long

        # Sort through segments since we don't want to remove primary segments
        secondary_objects, _ = segment_sort(skel_img, objects)

        # Keep segments longer than specified size
        for i in range(0, len(secondary_objects)):
            if len(secondary_objects[i]) > size:
                kept_segments.append(secondary_objects[i])
            else:
                removed_segments.append(secondary_objects[i])

        # Draw the contours that got removed
        removed_barbs = np.zeros(skel_img.shape[:2], np.uint8)
        cv2.drawContours(removed_barbs, removed_segments, -1, 255, 1,
                         lineType=8)

        # Subtract all short segments from the skeleton image
        pruned_img = image_subtract(pruned_img, removed_barbs)
        pruned_img = _iterative_prune(pruned_img, 1)

    # Reset debug mode
    params.debug = debug

    # Make debugging image
    if mask is None:
        pruned_plot = np.zeros(skel_img.shape[:2], np.uint8)
    else:
        pruned_plot = mask.copy()
    pruned_plot = cv2.cvtColor(pruned_plot, cv2.COLOR_GRAY2RGB)
    pruned_obj, _ = _cv2_findcontours(bin_img=pruned_img)
    cv2.drawContours(pruned_plot, removed_segments, -1, (0, 0, 255), params.line_thickness, lineType=8)
    cv2.drawContours(pruned_plot, pruned_obj, -1, (150, 150, 150), params.line_thickness, lineType=8)

    _debug(visual=pruned_img, filename=os.path.join(params.debug_outdir, f"{params.device}_pruned.png"))
    _debug(visual=pruned_img, filename=os.path.join(params.debug_outdir, f"{params.device}_pruned_debug.png"))

    # Segment the pruned skeleton
    segmented_img, segment_objects = segment_skeleton(pruned_img, mask)

    return pruned_img, segmented_img, segment_objects


#pruned_img, segmented_img, segment_objects = prune(mip_image_test, size=50, mask=None) #30 is fine 
#plot_images(pruned_img, mip_image_test, 'Processed', 'Original')

def pruneScenes(scenes, size=0, mask=None):
    """
    Apply the prune2D function to each 2D skeleton image (scene) in a list of scenes.
    
    Parameters:
        scenes (list): List of 2D skeletonized images (scenes).
        size (int): Size threshold to prune off segments.
        mask (ndarray): Optional binary mask for debugging.
    
    Returns:
        pruned_scenes (list): List of pruned 2D images for all scenes.
        segmented_scenes (list): List of segmented debugging images for all scenes.
        segment_objects_list (list): List of segment objects for all scenes.
    """
    pruned_scenes = []
    segmented_scenes = []
    segment_objects_list = []
    
    for i, scene in enumerate(scenes):
        print(f"Processing scene {i+1}/{len(scenes)}")
        
        # Apply pruning to the current scene using prune2D function
        pruned_img, segmented_img, segment_objects = prune2D(scene, size=size, mask=mask)
        
        # Append the results to the respective lists
        pruned_scenes.append(pruned_img)
        segmented_scenes.append(segmented_img)
        segment_objects_list.append(segment_objects)
    
    return pruned_scenes, segmented_scenes, segment_objects_list

# Example usage:
# pruned_scenes, segmented_scenes, segment_objects_list = pruneScenes(scenes, size=50, mask=None)




#Write image as tif. Ue imageJ for visualization
from skimage.io import imsave
#imsave("tttt.tif", tubeness_scenes[6]) 

import tifffile as tiff

def save_as_tiff(image_slice, file_name):
    """Save an image slice as a TIFF file."""
    tiff.imwrite(file_name, image_slice, photometric='minisblack')

# Example usage to save specific slices
#save_as_tiff(tubeness_scenes[6], 'tubeness_python.tif')
#save_as_tiff(blurred_scenes[2][8, :, :], 'blurred_scene_slice_8.tif')





# 3D prune 

from skimage.measure import label
from scipy.ndimage import distance_transform_edt

def prune3D(skel_img, size=0, mask=None):
    """Prune the ends of skeletonized segments in 3D."""
    
    pruned_img = skel_img.copy()

    # Label the connected components in 3D
    labeled_img, num_features = label(skel_img, connectivity=3, return_num=True)

    kept_segments = []
    removed_segments = []

    if size > 0:
        # Measure the size (e.g., length or volume) of each segment in 3D
        for segment_id in range(1, num_features + 1):
            segment = (labeled_img == segment_id)
            segment_size = np.sum(segment)

            if segment_size > size:
                kept_segments.append(segment)
            else:
                removed_segments.append(segment)

        # Subtract all short segments from the skeleton image
        for removed_segment in removed_segments:
            pruned_img[removed_segment] = 0

    # Optional: Create a debugging plot using a 3D projection or overlay on a mask
    if mask is not None:
        # Use 3D visualization tools like matplotlib or plotly for debugging
        pass
    
    # Return the pruned skeleton and optionally the segmented objects
    segmented_img, segment_objects = label(pruned_img, connectivity=3, return_num=True)

    return pruned_img, segmented_img, segment_objects


def prune3Dscenes(scenes, size=0, mask=None):
    """
    Apply pruning to each 3D numpy array in a list.
    
    Parameters:
        scenes (list): List of 3D numpy arrays where each array represents a scene.
        size (int): The minimum size of objects to keep.
        mask (numpy.ndarray): Optional mask for additional processing or visualization.
    
    Returns:
        list: A list of tuples, each containing pruned 3D numpy arrays, segmented images, and segment objects.
    """
    processed_scenes = []
    
    for i, scene in enumerate(scenes):
        print(f"Processing scene {i+1}/{len(scenes)}")
        
        if scene.size == 0:
            print(f"Scene {i+1} is empty or invalid!")
            continue
        
        try:
            pruned_img, segmented_img, segment_objects = prune3D(scene, size=size, mask=mask)
            processed_scenes.append((pruned_img, segmented_img, segment_objects))
            print(f"Processed scene {i+1} successfully.")
        except Exception as e:
            print(f"Error processing scene {i+1}: {e}")
    
    print(f"Total processed scenes: {len(processed_scenes)}")
    return processed_scenes


