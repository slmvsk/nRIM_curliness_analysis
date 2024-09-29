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
        img_uint8 = scene.astype(np.uint8) * 255

        # Apply pruning to the current scene using prune2D function
        pruned_img, segmented_img, segment_objects = prune2D(img_uint8, size=size, mask=mask)
        
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
#save_as_tiff(broken_skeleton, 'broken_skeleton.tif')
#save_as_tiff(blurred_scenes[2][8, :, :], 'blurred_scene_slice_8.tif')





# 3D prune 

from skimage.measure import label
from scipy.ndimage import distance_transform_edt

def prune3D(skel_img, size=0):
    """
    Prune the ends of skeletonized segments in 3D, removing small objects based on a size threshold.

    Parameters:
        skel_img (numpy.ndarray): A 3D binary numpy array representing the skeletonized image.
        size (int): Minimum size (e.g., length or volume) of segments to keep.

    Returns:
        numpy.ndarray: A 3D binary image with small objects removed.
    """
    pruned_img = skel_img.copy()

    # Label the connected components in 3D
    labeled_img, num_features = label(skel_img, connectivity=3, return_num=True)

    # Measure the size (e.g., length or volume) of each segment in 3D and remove small segments
    if size > 0:
        for segment_id in range(1, num_features + 1):
            segment = (labeled_img == segment_id)
            segment_size = np.sum(segment)

            if segment_size <= size:
                pruned_img[segment] = 0

    return pruned_img


def prune3Dscenes(scenes, size=0):
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
            pruned_img = prune3D(scene, size=size)
            processed_scenes.append(pruned_img)
            print(f"Processed scene {i+1} successfully.")
        except Exception as e:
            print(f"Error processing scene {i+1}: {e}")
    
    print(f"Total processed scenes: {len(processed_scenes)}")
    return processed_scenes



import numpy as np
from skimage.morphology import skeletonize
import scipy.ndimage
import matplotlib.pyplot as plt

def removeLoops(image):
    """
    Fill loops in the skeletonized image using binary_fill_holes and then re-skeletonize it.

    Parameters:
        image (numpy.ndarray): Input 2D binary skeleton image.

    Returns:
        numpy.ndarray: A re-skeletonized image after filling loops.
    """
    # Ensure the input image is binary
    binary_image = image > 0
    
    # Fill the holes using scipy's binary_fill_holes
    filled_image = scipy.ndimage.binary_fill_holes(binary_image)
    
    # Re-skeletonize the filled image
    skeletonized_image = skeletonize(filled_image)
    
    return skeletonized_image


# Apply loop removal
#skeleton_no_loops = removeLoops(pruned_skeleton)

def removeLoopsScenes(scenes):
    """
    Apply the removeLoops function to each 2D image in the list of scenes.

    Parameters:
        scenes (list): A list of 2D numpy arrays representing each scene.

    Returns:
        list: A list of 2D numpy arrays with loops removed and re-skeletonized.
    """
    # Initialize an empty list to store the processed scenes
    processed_scenes = []
    
    # Iterate through each 2D scene in the list
    for i, scene in enumerate(scenes):
        print(f"Processing scene {i+1}/{len(scenes)}")
        
        # Apply the removeLoops function to each 2D scene
        processed_scene = removeLoops(scene)
        
        # Append the processed scene to the list
        processed_scenes.append(processed_scene)
    
    return processed_scenes






# break remains branch pounts

"""Find branch points from skeleton image."""
import os
import cv2
import numpy as np
from plantcv.plantcv import params
from plantcv.plantcv import dilate
from plantcv.plantcv import outputs
from plantcv.plantcv._debug import _debug
from plantcv.plantcv._helpers import _cv2_findcontours


def find_branch_pts(skel_img, mask=None, label=None):
    """Find branch points in a skeletonized image.
    The branching algorithm was inspired by Jean-Patrick Pommier: https://gist.github.com/jeanpat/5712699

    Inputs:
    skel_img    = Skeletonized image
    mask        = (Optional) binary mask for debugging. If provided, debug image will be overlaid on the mask.
    label        = Optional label parameter, modifies the variable name of
                   observations recorded (default = pcv.params.sample_label).

    Returns:
    branch_pts_img = Image with just branch points, rest 0

    :param skel_img: numpy.ndarray
    :param mask: np.ndarray
    :param label: str
    :return branch_pts_img: numpy.ndarray
    """
    # Set lable to params.sample_label if None
    if label is None:
        label = params.sample_label

    # In a kernel: 1 values line up with 255s, -1s line up with 0s, and 0s correspond to don't care
    # T like branch points
    t1 = np.array([[-1, 1, -1],
                   [1, 1, 1],
                   [-1, -1, -1]])
    t2 = np.array([[1, -1, 1],
                   [-1, 1, -1],
                   [1, -1, -1]])
    t3 = np.rot90(t1)
    t4 = np.rot90(t2)
    t5 = np.rot90(t3)
    t6 = np.rot90(t4)
    t7 = np.rot90(t5)
    t8 = np.rot90(t6)

    # Y like branch points
    y1 = np.array([[1, -1, 1],
                   [0, 1, 0],
                   [0, 1, 0]])
    y2 = np.array([[-1, 1, -1],
                   [1, 1, 0],
                   [-1, 0, 1]])
    y3 = np.rot90(y1)
    y4 = np.rot90(y2)
    y5 = np.rot90(y3)
    y6 = np.rot90(y4)
    y7 = np.rot90(y5)
    y8 = np.rot90(y6)
    kernels = [t1, t2, t3, t4, t5, t6, t7, t8, y1, y2, y3, y4, y5, y6, y7, y8]

    branch_pts_img = np.zeros(skel_img.shape[:2], dtype=int)

    # Store branch points
    for kernel in kernels:
        branch_pts_img = np.logical_or(cv2.morphologyEx(skel_img, op=cv2.MORPH_HITMISS, kernel=kernel,
                                                        borderType=cv2.BORDER_CONSTANT, borderValue=0), branch_pts_img)

    # Switch type to uint8 rather than bool
    branch_pts_img = branch_pts_img.astype(np.uint8) * 255

    # Store debug
    debug = params.debug
    params.debug = None

    # Make debugging image
    if mask is None:
        dilated_skel = dilate(skel_img, params.line_thickness, 1)
        branch_plot = cv2.cvtColor(dilated_skel, cv2.COLOR_GRAY2RGB)
    else:
        # Make debugging image on mask
        mask_copy = mask.copy()
        branch_plot = cv2.cvtColor(mask_copy, cv2.COLOR_GRAY2RGB)
        skel_obj, skel_hier = _cv2_findcontours(bin_img=skel_img)
        cv2.drawContours(branch_plot, skel_obj, -1, (150, 150, 150), params.line_thickness, lineType=8,
                         hierarchy=skel_hier)

    branch_objects, _ = _cv2_findcontours(bin_img=branch_pts_img)

    # Initialize list of tip data points
    branch_list = []
    branch_labels = []
    for i, branch in enumerate(branch_objects):
        x, y = branch.ravel()[:2]
        coord = (int(x), int(y))
        branch_list.append(coord)
        branch_labels.append(i)
        cv2.circle(branch_plot, (x, y), params.line_thickness, (255, 0, 255), -1)

    outputs.add_observation(sample=label, variable='branch_pts',
                            trait='list of branch-point coordinates identified from a skeleton',
                            method='plantcv.plantcv.morphology.find_branch_pts', scale='pixels', datatype=list,
                            value=branch_list, label=branch_labels)

    # Reset debug mode
    params.debug = debug

    _debug(visual=branch_plot, filename=os.path.join(params.debug_outdir, f"{params.device}_branch_pts.png"))

    return branch_pts_img




def break_at_junctions(skel_img, branch_points):
    """
    Break the skeleton at the identified junction points.
    
    Parameters:
        skel_img (numpy.ndarray): A 2D binary skeleton image.
        branch_points (numpy.ndarray): Binary image with junction points (branch points) marked.
    
    Returns:
        numpy.ndarray: A skeleton image with junction points removed (broken skeleton).
    """
    broken_skeleton = skel_img.copy()
    
    # Remove junction points from the skeleton to break complex intersections
    broken_skeleton[branch_points > 0] = 0
    
    return broken_skeleton





import numpy as np
from skimage.morphology import skeletonize, thin
from skimage.measure import label
from skimage.morphology import remove_small_objects
import cv2
from skimage.color import label2rgb
from skimage.measure import label


#def breakJunctionsAndLabelScenes(scenes, num_iterations=3):

    #colored_skeletons = []

    #for i, scene in enumerate(scenes):
        #print(f"Processing scene {i + 1}/{len(scenes)}")
        
        #try:
            # Make a copy of the scene to process
            #broken_skel = scene.copy()

            # Iterate to break junctions multiple times
            #for _ in range(num_iterations):
                #branch_points = find_branch_pts(broken_skel)
                #broken_skel = break_at_junctions(broken_skel, branch_points)

            # Label connected components in the broken skeleton
            #labeled_skel = label(broken_skel, connectivity=2)

            # Colorize the labeled skeleton (each label gets a different color)
            #colored_skel = label2rgb(labeled_skel, bg_label=0)
            #colored_skel = broken_skel

            # Append the colored skeleton to the list
            #colored_skeletons.append(colored_skel)

        #except Exception as e:
            #print(f"Error processing scene {i + 1}: {e}")
            #continue

    #return colored_skeletons


def breakJunctionsAndLabelScenes(scenes, num_iterations=3):
    """
    Iterate over all scenes in a list, break skeletons at junctions, and label each separate branch with a different color.

    Parameters:
        scenes (list): List of 2D skeleton images.
        num_iterations (int): Number of iterations to break at junctions.

    Returns:
        colored_skeletons (list): List of skeleton images with separate branches color-labeled.
    """
    colored_skeletons = []

    for i, scene in enumerate(scenes):
        print(f"Processing scene {i + 1}/{len(scenes)}")
        
        try:
            # Ensure the input skeleton is in uint8 format
            broken_skel = (scene > 0).astype(np.uint8)

            # Iterate to break junctions multiple times
            for _ in range(num_iterations):
                branch_points = find_branch_pts(broken_skel)  # Assuming this function returns branch points as binary
                branch_points_uint8 = (branch_points > 0).astype(np.uint8)  # Ensure branch points are uint8
                broken_skel = break_at_junctions(broken_skel, branch_points_uint8)

            # Label connected components in the broken skeleton
            labeled_skel = label(broken_skel, connectivity=2)

            # Colorize the labeled skeleton (each label gets a different color)
            colored_skel = label2rgb(labeled_skel, bg_label=0, kind='avg')

            # Append the colored skeleton to the list
            colored_skeletons.append(colored_skel)

        except Exception as e:
            print(f"Error processing scene {i + 1}: {e}")
            continue

    return colored_skeletons




