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
    # Convert to 8 bit (no need for now)
    
    # Step 2: Normalize intensity 
    normalized_scenes = normalizeScenes(scenes)
    del scenes  # Free memory used by the original scenes
    
    # Step 3: Remove soma
    nosoma_scenes = removeSomaFromAllScenes(normalized_scenes, xy_resolution=1)
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
    skeletonized_scenes = process_scenes_for_skeletonization(tubeness_scenes)
    del tubeness_scenes  # Free memory used by the tubeness scenes
    
    # Step 6: Z-projection
    mip_scenes = do_mip_scenes(skeletonized_scenes)
    del skeletonized_scenes  # Free memory used by the clean skeleton scenes
    
    # Step 7: Clean skeleton
    dendrite_lengths_scenes = measure_branch_lengths_batch(mip_scenes)
    # You can now calculate min, max, and mean for each scene
    scene_stats = [(np.min(lengths), np.max(lengths), np.mean(lengths)) for lengths in all_lengths]
    for idx, (min_len, max_len, mean_len) in enumerate(scene_stats):
        print(f"Scene {idx+1} - Min: {min_len}, Max: {max_len}, Mean: {mean_len}")
    # based on these measurments function will clean skeleton
    # make comment if you don't want to see if there were branches and whats their measure 
    cleaned_scenes = cleanMipSkeleton(mip_scenes, length_percentiles=(70, 100))
    del mip_scenes  # Free memory used by the skeleton scenes
    
    print(f"Finished processing file: {file_name}")
    
    return cleaned_scenes








import pandas as pd

def batch_process_files(file_list, process_function, folder_path):
    """
    Batch processes files and compiles results into a pandas DataFrame.
    
    Args:
    file_list (list): List of file names to be processed.
    process_function (function): Function to process each file.
    folder_path (str): Path to the folder containing the files.
    
    Returns:
    pd.DataFrame: DataFrame containing results from all processed files.
    """
    # List to store DataFrame rows before concatenation
    all_rows = []

    for file_name in file_list:
        # Construct full file path
        file_path = os.path.join(folder_path, file_name)
        
        # Process the file
        cleaned_scenes = processFile(file_path)
        
        # Collect each scene's data in all_rows list
        for scene_index, cleaned_scene in enumerate(cleaned_scenes):
            all_rows.append({
                "file_name": file_name,
                "scene_index": scene_index,
                "cleaned_scene": cleaned_scene
            })
        
        print(f"Finished processing {file_name}")
    
    # Convert list of dicts to DataFrame
    results_df = pd.DataFrame(all_rows)
    
    return results_df

# Example usage
folder_with_data = '/Users/tetianasalamovska/Desktop/zeis'
file_list = getMatchingFilesList(
    folder_with_data,
    EXPERIMENT='IHCT',
    MAGN_LIST=['40x2x', '40x3x'],
    ID='THT53',
    SEPARATOR='_',
    EXTENSION='.czi'
)


dataframe_results = batch_process_files(file_list, processFile, folder_with_data)
 



def release_memory(variable):
    del variable







    
    


