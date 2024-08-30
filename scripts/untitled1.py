#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 04:49:20 2024

@author: tetianasalamovska
"""

# import 'scenes' done 

# normalize scenes 



plot_image_histogram(scenes[6][15,:,:])

normalized_scenes = normalizeScenes(scenes)

plot_images(normalized_scenes[6][17,:,:], scenes[6][17,:,:], 'nm', 'orig')

plot_image_histogram(normalized_scenes[6][15,:,:])

# histogram with peak in the end  as before but stretched 

# apply gaussian blur 

blurred_scenes = apply_gaussian(normalized_scenes, sigma=1) # try sigma 1 

plot_image_histogram(blurred_scenes[8][15,:,:]) #peaks in the end removed 

plot_images(normalized_scenes[8][10,:,:], blurred_scenes[8][10,:,:], 'nm', 'blurr')

# so far so good 

# threshol 
thresholds = 0.25 #binarising
nosoma_scenes = removeSomaFromAllScenes(blurred_scenes, thresholds) 


unique_values = np.unique(nosoma_scenes[6])
print("Unique values in the binary image:", unique_values)

plot_images(nosoma_scenes[8][10,:,:], blurred_scenes[8][10,:,:], 'nm', 'blurr')



#trying closing (do after cleanin!!!!)
radius_value = 4  # Adjust the radius value as needed
closed_scene = apply_closing(cleaned_nosoma, radius=radius_value)
plot_images(nosoma_scenes[8][10,:,:], closed_scene[10,:,:], 'Processed', 'Original')






# cleaning
cleaned_nosoma = remove_small_objects_3d(cleaned_scenes[8], min_size=5000)
plot_images(cleaned_nosoma[10,:,:], nosoma_scenes[8][10,:,:], 'nosoma', 'closed')


cleaned_scenes = process_scenes(nosoma_scenes, min_size=2000)
plot_images(cleaned_scenes[8][10,:,:], nosoma_scenes[8][10,:,:], 'Processed', 'Original')

unique_values = np.unique(cleaned_scenes[6])
print("Unique values in the binary image:", unique_values)


plot_images(cleaned_scenes[6][18,:,:], nosoma_scenes[6][18,:,:], 'Processed', 'Original')
#!!!!!!!!! too agressive here 
# looks like it is fine to do closing or similar operation after thresholding and before cleaning
# to connct some pixels 
# and do less agressive cleaning

#median filter is the solution!
from scipy.ndimage import median_filter
smoothed_image = median_filter(closed_scene, size=8)  # Adjust size as needed

plot_images(closed_scene[10,:,:], smoothed_image[10,:,:], 'Processed', 'Original')


#opening ?
#skeletonize
skeletonized = skeletonize_image(cleaned_nosoma)
skeletonized_test = skeletonize(cleaned_scenes[8])
plot_images(skeletonized[18,:,:], labeled_skeleton[18,:,:], 'Processed', 'Original')

# mip to check or save 
save_as_tiff(skeletonized, 'skeletonized_test.tiff') 

# good enough with this example 

#treat as 3D skeleton for further analysis 
# clean skeleton !!!!!!!!
cleaned_skeleton = #remove_small_objects_3d(skeletonized, min_size=10)
min_branch_length = 200  # 200 is fine 
cleaned_skeleton = clean_skeleton_3d(skeletonized, min_length=min_branch_length)
save_as_tiff(pruned, 'pruned.tiff') 


mip_image_test = max_intensity_z_projection(skeletonized_test)
plot_images(skeletonized[10,:,:], mip_image_test, 'Processed', 'Original')


# pruning of small branches 
# Prune barbs off skeleton image

import os
import cv2
import numpy as np
from plantcv.plantcv import params
from plantcv.plantcv import image_subtract
from plantcv.plantcv.morphology import segment_sort
from plantcv.plantcv.morphology import segment_skeleton
from plantcv.plantcv.morphology import _iterative_prune
from plantcv.plantcv._debug import _debug
from plantcv.plantcv._helpers import _cv2_findcontours


def prune(skel_img, size=0, mask=None):
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


pruned_img, segmented_img, segment_objects = prune(mip_image_test, size=50, mask=None) #30 is fine 
plot_images(pruned_img, mip_image_test, 'Processed', 'Original')



# 3D function 







# Example usage:
# Assuming `mip_image_test_3d` is your 3D skeletonized image
pruned_img, kept_segments, removed_segments = prune_3d(skeletonized, size=50)



pruned3dmip = max_intensity_z_projection(pruned_img)
plot_images(pruned3dmip, mip_image_test, 'Processed', 'Original')





# try to do morphological opening before measuring curliness 




#######check my original skeletonize function for scenes and continue doing for scenes 
#and check all of them 

# how to remove small branches from large branches
# opening in 3D skeleton? forst try without it just clean 
# or proceed to analysis with 3D skeleton and label and see 










