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



# Cleaning skeleton in 3D (doesnt work)

def cleanSkeleton3d(skeleton_image, min_length=10):
    """
    Measure and remove branches shorter than a specified length in a 3D skeletonized image.

    Parameters:
        skeleton_image (numpy.ndarray): A 3D binary numpy array representing the skeleton (values 0 and 1).
        min_length (float): The minimum length of branches to keep.

    Returns:
        numpy.ndarray: A 3D binary image with small branches removed.
    """
    # Label the skeletonized image to identify connected components (branches)
    labeled_skeleton = label(skeleton_image, connectivity=3)
    props = regionprops(labeled_skeleton)

    # Create an empty array to hold the cleaned skeleton
    cleaned_skeleton = np.zeros_like(skeleton_image)

    # Iterate over each labeled component (branch)
    for prop in props:
        coords = prop.coords
        if len(coords) < 2:
            continue  # Skip if the branch has fewer than 2 pixels
        
        # Measure the branch length
        branch_length = 0
        for i in range(len(coords) - 1):
            branch_length += euclidean(coords[i], coords[i + 1])
        
        # Keep the branch if it's longer than the minimum length
        if branch_length >= min_length:
            for coord in coords:
                cleaned_skeleton[tuple(coord)] = 1
    
    return  cleaned_skeleton













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

def measure_branch_lengths_batch(scenes_2d):
    """
    Measure the branch lengths for a list of 2D binary skeletonized images.

    Args:
    scenes_2d (list): A list of 2D binary images of skeletonized structures.

    Returns:
    list: A list of lists, where each sublist contains the lengths of branches in the corresponding scene.
    """
    all_lengths = []
    for idx, scene in enumerate(scenes_2d):
        lengths = []
        skeleton = scene > 0
        labeled_skeleton = label(skeleton)
        props = regionprops(labeled_skeleton)

        for prop in props:
            coords = prop.coords
            if len(coords) < 2:
                continue  # Skip if the branch has fewer than 2 pixels
            branch_length = 0
            for i in range(len(coords) - 1):
                branch_length += euclidean(coords[i], coords[i + 1])
            lengths.append(branch_length)
        
        all_lengths.append(lengths)
        print(f"Processed scene {idx+1}/{len(scenes_2d)}: {len(lengths)} branches measured")
    
    return all_lengths

def cleanMipSkeleton(scenes_2d, length_percentiles=(5, 95)):
    """
    Clean a list of 2D binary skeleton images by removing small or excessively large branches based on dynamic length thresholds.

    Args:
    scenes_2d (list): A list of 2D binary images of skeletonized structures.
    length_percentiles (tuple): Percentiles to determine the min and max branch lengths to keep (default is (5, 95)).

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
        
        lengths = []
        for prop in props:
            coords = prop.coords
            branch_length = 0
            for i in range(len(coords) - 1):
                branch_length += euclidean(coords[i], coords[i + 1])
            lengths.append(branch_length)
        
        if len(lengths) > 0:
            # Set dynamic min and max lengths based on percentiles
            min_length = np.percentile(lengths, length_percentiles[0])
            max_length = np.percentile(lengths, length_percentiles[1])
        else:
            min_length = 0
            max_length = float('inf')
        
        print(f"Scene {idx+1} - Min Length: {min_length:.2f}, Max Length: {max_length:.2f}")
        
        # Create a new cleaned skeleton image
        cleaned_skeleton = np.zeros_like(binary_image)
        
        for prop in props:
            coords = prop.coords
            branch_length = 0
            for i in range(len(coords) - 1):
                branch_length += euclidean(coords[i], coords[i + 1])
            
            # Keep branches that meet the dynamic length criteria
            if min_length <= branch_length <= max_length:
                for coord in coords:
                    cleaned_skeleton[coord[0], coord[1]] = 1  # Set pixel to 1 to keep the branch
        
        cleaned_scenes.append(cleaned_skeleton)
        print(f"Processed scene {idx+1}/{len(scenes_2d)}: Skeleton cleaned")
    
    return cleaned_scenes



#######################################
# Prune 


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





