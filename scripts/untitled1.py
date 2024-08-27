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

plot_images(normalized_scenes[6][17,:,:], blurred_scenes[6][17,:,:], 'nm', 'blurr')

# so far so good 

# threshol 
thresholds = 0.25 #binarising
nosoma_scenes = removeSomaFromAllScenes(blurred_scenes, thresholds) 


unique_values = np.unique(nosoma_scenes[6])
print("Unique values in the binary image:", unique_values)

plot_images(nosoma_scenes[6][17,:,:], blurred_scenes[6][17,:,:], 'nm', 'blurr')



#trying closing 
radius_value = 3  # Adjust the radius value as needed
closed_scene = apply_closing(nosoma_scenes[6], radius=radius_value)
plot_images(nosoma_scenes[6][18,:,:], closed_scene[18,:,:], 'Processed', 'Original')






# cleaning
cleaned_nosoma = remove_small_objects_3d(closed_scene, min_size=2000)
plot_images(cleaned_nosoma[18,:,:], closed_scene[18,:,:], 'nosoma', 'closed')


cleaned_scenes = process_scenes(nosoma_scenes, min_size=2000)
plot_images(cleaned_scenes[7][18,:,:], nosoma_scenes[7][18,:,:], 'Processed', 'Original')

unique_values = np.unique(cleaned_scenes[6])
print("Unique values in the binary image:", unique_values)


plot_images(cleaned_scenes[6][18,:,:], nosoma_scenes[6][18,:,:], 'Processed', 'Original')
#!!!!!!!!! too agressive here 
plot_images(cleaned_scenes[6][18,:,:], scenes[6][18,:,:], 'Processed', 'Original')
# looks like it is fine to do closing or similar operation after thresholding and before cleaning
# to connct some pixels 
# and do less agressive cleaning

#opening ?
#skeletonize
skeletonized = skeletonize_image(cleaned_nosoma)
skeletonized_test = skeletonize(cleaned_nosoma)
plot_images(skeletonized[18,:,:], skeletonized_test[18,:,:], 'Processed', 'Original')

# mip to check or save 
save_as_tiff(skeletonized_test, 'skeletonized_test.tiff') 

# good enough with this example 

#treat as 3D skeleton for further analysis 





# do closing later or with skeleton? 





