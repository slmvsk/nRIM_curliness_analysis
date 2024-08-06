#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 14:29:08 2024

@author: tetianasalamovska
"""
#IN PROGRESS 

def processFile(file_name):
    # Step 1: Read the file as a list of 3D numpy arrays (scenes)
    scenes, metadata = readCziFile(file_name)
    
    
    #Step 1.5 for memory efficiency:
    # Convert to 8 bit
    
    
    # Step 2: Normalize intensity 
    normalized_scenes = normalizeScenes(scenes)
    del scenes  # Free memory used by the original scenes
    
    # Step 3: Remove soma
    nosoma_scenes = removeSomaFromScenes(normalized_scenes, xy_resolution=1)
    # validate (just plotting)
    plot_images(normalized_scenes[8][1,:,:], nosoma_scenes[8][1,:,:], 'Original', 'No Somata')
    #del normalized_scenes  # Free memory used by the normalized scenes
    
    # Step 4: Apply tubeness filter
    # sigma = 
    tubeness_scenes = tubenessForAllScenes(nosoma_scenes, scale_factor=0.9)
    del nosoma_scenes  # Free memory used by the nosoma scenes
    
    #Step 4.1: Validate tubeness
    #validation = subtract_tubeness_from_nosoma(nosoma_scenes[8], tubeness_scenes[8])
    #plot_images(tubeness_scenes[8][8,:,:], nosoma_scenes[8][8,:,:], 'Tubeness', 'No Somata')
    #plot_images(validation[8,:,:], nosoma_scenes[8][8,:,:], 'Substracted', 'No Somata')
    # Inside the function can be changed to see the opposite substraction
    
    # Step 5: Binarize and skeletonize
    skeleton_scenes = binarize_and_skeletonize(tubeness_scenes)
    del tubeness_scenes  # Free memory used by the tubeness scenes
    
    # Step 6: Clean skeleton
    clean_skeleton_scenes = clean_skeleton(skeleton_scenes)
    del skeleton_scenes  # Free memory used by the skeleton scenes
    
    # Step 7: Z-projection
    z_projected_skeletons = z_project(clean_skeleton_scenes)
    del clean_skeleton_scenes  # Free memory used by the clean skeleton scenes
    
    # Step 8: Store or return the final result
    store_skeleton(z_projected_skeletons, file_name)
    del z_projected_skeletons  # Free memory used by the final result

    print(f"Finished processing file: {file_name}")
    





def release_memory(variable):
    del variable







    
    


