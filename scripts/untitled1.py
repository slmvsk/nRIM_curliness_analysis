#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 04:49:20 2024

@author: tetianasalamovska
"""

# import 'scenes' done 

# normalize scenes 




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










