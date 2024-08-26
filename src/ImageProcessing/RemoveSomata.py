#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 00:40:39 2024

@author: tetianasalamovska
"""

# thresholding = denoising! Otsu?  ####### make 8 bit before this step 
##############################################

# Operations like enhancing contrast or thresholding can be done as for 2D 
# images where the context of adjacent slices isn’t important ? 
# Segmentation must be done for scenes not for slices 

import numpy as np
from skimage.filters import threshold_multiotsu
from skimage import img_as_float
import matplotlib.pyplot as plt



# Manually determined threshold based on histogram analysis
threshold_values = [0.2, 0.6]  # Example thresholds

# Initialize a segmented stack
segmented_stack = np.zeros_like(test_img)

# Apply thresholds to each slice
for i in range(test_img.shape[0]):
    img_float = img_as_float(test_img[i, :, :])
    segmented_stack[i, :, :] = np.digitize(img_float, bins=threshold_values)

# Optionally visualize one of the segmented slices to verify the segmentation
plt.figure(figsize=(10, 4))
plt.imshow(segmented_stack[slice_index, :, :], cmap='gray')
plt.title('Segmented Middle Slice')
plt.axis('off')
plt.show()

# Optionally visualize one of the segmented slices to verify the segmentation
plt.figure(figsize=(10, 4))
plt.imshow(test_img[10, :, :], cmap='gray')
plt.title('Segmented Middle Slice')
plt.axis('off')
plt.show()

################## putting all above to the function again #######################

# use this function for now for good segmentation 

def apply_thresholds_to_stack(image_stack, thresholds):
    """
    Apply given threshold values to segment a 3D image stack.
    
    Parameters:
        image_stack (ndarray): A 3D numpy array representing the image stack.
        thresholds (list): A list of threshold values to segment the image.
    
    Returns:
        ndarray: A 3D numpy array of the segmented image stack.
    """
    # Initialize a segmented stack
    segmented_stack = np.zeros_like(image_stack)
    
    # Apply thresholds to each slice
    for i in range(image_stack.shape[0]):
        img_float = img_as_float(image_stack[i, :, :])
        segmented_stack[i, :, :] = np.digitize(img_float, bins=thresholds)
    
    return segmented_stack

# Example usage
thresholds = [0.22, 0.45]  # Example thresholds
segmented_stack = apply_thresholds_to_stack(normalized_scenes[9], threshold_values)

#debugging step 
def removeSomaFromAllScenes(scenes, thresholds):
    """
    Iterate over all scenes in a file, apply the removeSomafromStack function to each scene,
    and release memory after processing each scene.
    
    Parameters:
        scenes (list): List of 3D numpy arrays where each array represents a scene.
        xy_resolution (float): Resolution scaling factor in the XY plane.
    
    Returns:
        list: A list of 3D numpy arrays with somas removed.
    """
    processed_scenes = []
    
    for i, scene in enumerate(scenes):
        print(f"Processing scene {i+1}/{len(scenes)}")
        
        # Check if scene is valid
        if scene.size == 0:
            print(f"Scene {i+1} is empty or invalid!")
            continue
        
        try:
            # Apply the removeSomafromStack function to the current scene
            processed_scene = apply_thresholds_to_stack(scene, thresholds) #changed to manual temporary 
            processed_scenes.append(processed_scene)
            print(f"Processed scene {i+1} successfully added to the list.")
              
        except Exception as e:
            print(f"Error processing scene {i+1}: {e}")
            continue
        # Release memory for the current scene
        del scene
    print(f"Total processed scenes: {len(processed_scenes)}")
    return processed_scenes


nosoma_scenes = removeSomaFromAllScenes(normalized_scenes, thresholds)
print(f"Number of scenes processed and returned: {len(nosoma_scenes)}")

plot_images(normalized_scenes[9][8,:,:], segmented_stack[8,:,:], 'Original', 'No soma')
# fine enough 









# Optionally, inspect the first scene to ensure it's not empty
if len(nosoma_scenes) > 0:
    print(f"Shape of the first processed scene: {nosoma_scenes[0].shape}")
else:
    print("No scenes were processed.")
    
plot_images(binary_image[8,:,:], nosoma_scenes[9][8,:,:], 'Original', 'No soma')



# code USED BEFORE FOR AUTOMATIC THRESHOLDING (finding optimal levels) 
def findOptimalThreshold(img, metric_th=0.95):
    """Determine the optimal number of threshold levels based on a target metric threshold."""
    metrics = []
    optimal_th = 1
    for th_lvl in range(1, 11):  # Test from 1 to 10 levels
        thresholds = threshold_multiotsu(img, classes=th_lvl)
        # Calculate a metric for these thresholds; here we use a simple placeholder
        # In practice, you'd want a metric that evaluates segmentation quality
        metric = np.var(thresholds) / np.mean(thresholds)
        metrics.append(metric)
        if metric > metric_th:
            optimal_th = th_lvl
            break
    else:
        # If no threshold level meets the threshold metric, pick the one with the highest metric
        optimal_th = np.argmax(metrics) + 1
    return optimal_th
def removeSomafromStack(image_stack, xy_resolution):
    """Remove somas from an image stack based on intensity thresholds."""
    img_float = img_as_float(image_stack)  # Ensure the image is in floating point
    n_slices = image_stack.shape[2]
    th_lvl = findOptimalThreshold(image_stack[:, :, n_slices // 2])
    
    # Apply multi-level thresholding
    thresholds = threshold_multiotsu(img_float[:, :, n_slices // 2], classes=th_lvl)
    quant_a = np.digitize(img_float, bins=thresholds)
    
    # Create background mask
    bg_mask = quant_a <= th_lvl * 0.2 # * 0.3 is fine 
    
    # Filter image stack: set background regions to zero
    image_stack_filtered = np.copy(image_stack)
    for i in range(n_slices):
        image_stack_filtered[:, :, i][bg_mask[:, :, i]] = 0

    return image_stack_filtered
nosoma_stack = removeSomafromStack(equalized_image, xy_resolution=1)
#img_filtered = median(normalized_scenes[8], ball(3))  # ball(2) provides a reasonable balance in 3D
#nosoma_img_med = removeSomafromStack(img_filtered, xy_resolution=1)
plot_images(normalized_scenes[4][8,:,:], nosoma_img[8,:,:], 'Original', 'No soma')
 
    



### NEXT STEP IS TO REMOVE SMALL OBJECTS ################################
# adapting clean skeleton function here before skeletonizing and tubeness 

# this one of the optoions ( number 2nd priority)
import matplotlib.pyplot as plt
from skimage import data
from skimage import color, morphology

footprint = morphology.disk(2)
res = morphology.white_tophat(nosoma_scenes[4][8,:,:], footprint)

fig, ax = plt.subplots(ncols=3, figsize=(20, 8))
ax[0].set_title('Original')
ax[0].imshow(nosoma_scenes[4][8,:,:], cmap='gray')
ax[1].set_title('White tophat')
ax[1].imshow(res, cmap='gray')
ax[2].set_title('Complementary')
ax[2].imshow(nosoma_scenes[4][8,:,:] - res, cmap='gray')
plt.show()
# Display the original image
plt.figure(figsize=(10, 10))  # Large display size
plt.imshow(nosoma_scenes[4][8,:,:], cmap='gray')
plt.title('Original')
plt.axis('off')  # Hide the axes
plt.show()
# Display the white tophat transformed image
plt.figure(figsize=(10, 10))  # Large display size
plt.imshow(res, cmap='gray')
plt.title('White Tophat')
plt.axis('off')  # Hide the axes
plt.show()
# Display the complementary image
complementary = nosoma_scenes[4][8,:,:] - res
plt.figure(figsize=(10, 10))  # Large display size
plt.imshow(complementary, cmap='gray')
plt.title('Complementary')
plt.axis('off')  # Hide the axes
plt.show()

# complementary is fine result but now i need to remove small objects literaly 


# ignore for now (good option)
#Area opening removes all connected components (clusters of pixels) that have fewer pixels than a specified threshold. Unlike a standard opening, which defines structure by shape, area opening targets small objects based on size.
#from skimage.morphology import area_opening
# Apply area opening ## seems better 
#cleaned_image = area_opening(nosoma_scenes[4][8,:,:], area_threshold=100)  # Adjust the threshold as needed
#plt.figure(figsize=(10, 10))  # Large display size
#plt.imshow(cleaned_image, cmap='gray')
#plt.title('?')
#plt.axis('off')  # Hide the axes
#plt.show()
#try to apply all of that before removing soma and thresholding 


#another method 
from skimage import morphology, measure, filters
from skan import csr

from skimage.measure import label, regionprops
import numpy as np
import matplotlib.pyplot as plt
#########HERE FIX

#i have 3nary image, convert to binary 
import numpy as np
from skimage.measure import label, regionprops
from skimage import morphology


# frstly remove small obj and then binarise?
import numpy as np

def convert_to_binary(image):
    """
    Convert an image with values 0, 1, 2 to a binary image with values 0, 1.

    Parameters:
        image (numpy.ndarray): A numpy array representing the image.

    Returns:
        numpy.ndarray: A binary image with values 0, 1.
    """
    binary_image = np.where(image > 0, 1, 0)
    return binary_image

# Example usage
# Assuming `image` is your numpy array with values 0, 1, 2
binary_image = convert_to_binary(nosoma_scenes[9])




def remove_small_objects_3d(binary_image, min_size=50):
    """
    Remove small objects from a 3D binary image (values 0 and 1) based on their size.

    Parameters:
        binary_image (numpy.ndarray): A 3D binary numpy array with values 0 and 1.
        min_size (int): The minimum size of objects to keep.

    Returns:
        numpy.ndarray: A 3D binary image with small objects removed.
    """
    # Label the binary image
    labeled_image = label(binary_image, connectivity=3)

    # Remove small objects
    cleaned_image = morphology.remove_small_objects(labeled_image, min_size=min_size, connectivity=3)
    
    # Convert the cleaned labeled image back to binary
    binary_cleaned_image = np.where(cleaned_image > 0, 1, 0)
    
    return binary_cleaned_image

# this is 3d connectivity 

# Usage example
cleaned_nosoma = remove_small_objects_3d(binary_image, min_size=10000)

plt.figure(figsize=(10, 10))  # Large display size
plt.imshow(cleaned_nosoma[8,:,:], cmap='gray')
plt.title('Cleaned Image')
plt.axis('off')  # Hide the axes
plt.show()

cleaned_nosoma = remove_small_objects_3d(nosoma_scenes[4], min_size=1)

plt.figure(figsize=(10, 10))  # Large display size
plt.imshow(cleaned_nosoma[8,:,:], cmap='gray')
plt.title('?')
plt.axis('off')  # Hide the axes
plt.show()


import matplotlib.pyplot as plt
from skimage import exposure, io
from skimage.filters import threshold_otsu

# Load an example image

# Display the histogram of the image
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(normalized_scenes[4][8,:,:], cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

# Histogram and threshold line
hist, bins_center = exposure.histogram(normalized_scenes[4][8,:,:])
ax[1].plot(bins_center, hist, lw=2)
ax[1].set_title('Histogram of Pixel Intensities')

# Apply Otsu's method to find an optimal threshold
thresh = threshold_otsu(normalized_scenes[4][8,:,:])
ax[1].axvline(thresh, color='r', ls='--')

ax[1].text(thresh+0.02, max(hist)/2, f'Threshold: {thresh}', color='red')

# Apply threshold
binary_image = image > thresh
fig, ax2 = plt.subplots(figsize=(6, 6))
ax2.imshow(binary_image, cmap='gray')
ax2.set_title('Binary Image After Thresholding')
ax2.axis('off')

plt.show()




















####################################
# yes it is for 0ne scene only, but all my functions will be for 1 scene and then 
# I will just iterate over all of the scenes and files?????????????????
#or make a function that will iterates for files but not scenes, that will 
# be already made for scenes 
# it is also a question that i will release memory after each scene or after each file?
####################################


# Example usage
# Assume `image_stack` is your 3D numpy array with shape [height, width, depth]
# `xy_resolution` is a parameter that you might use to adjust algorithm behavior based on image resolution


plot_images(normalized_scenes[5][8,:,:], nosoma_img[8,:,:], 'Original', 'No soma')



# geometrical approach or deep learningg approach to remove leftovers 

# for my example i need boinarise manually
# try Z-projection 
#mip_image = max_intensity_z_projection(image_nosoma)

from skimage import restoration, exposure


# Applying contrast stretching
p2, p98 = np.percentile(nosoma_img, (0.5, 99.5))
contrast_stretched = exposure.rescale_intensity(nosoma_img, in_range=(p2, p98))



