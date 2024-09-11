#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 14:29:08 2024

@author: tetianasalamovska
"""

import scipy
import gc



def processFile(file_name):
    # Step 1: Read the file as a list of 3D numpy arrays (scenes)
    scenes, metadata = readCziFile(file_name)
    
    # Step 2: Normalize intensity 
    normalized_scenes = normalizeScenes(scenes, percentiles=[1,99])
    del scenes  # Free memory used by the original scenes
    
    # Step 3: Denoising 
    blurred_scenes = applyGaussian(normalized_scenes, sigma=2)
    del normalized_scenes
    # Step 4: Background substraction and contrast enhancement 
    subtracted_scenes = subtractBackgroundFromScenes(blurred_scenes, radius=25)
    del blurred_scenes
    
    median_scenes = applyMedianFilter(subtracted_scenes, size=3)
    del subtracted_scenes
    
    stretched_scenes = applyContrastStretching(median_scenes, lower_percentile=1, upper_percentile=99)
    del median_scenes
    
    # Step 5: Remove soma
    binary_scenes = otsuThresholdingScenes(stretched_scenes) # maybe put old nosoma adaptive thresholding function back here 
    del stretched_scenes
    
    # Step 6: Cleaning
    cleaned_scenes = cleanBinaryScenes(binary_scenes, min_size=4000) 
    del binary_scenes
    
    eroded_scenes = applyErosionToScenes(cleaned_scenes, iterations=2, structure=np.ones((3, 3, 3)))
    del cleaned_scenes
    
    dilated_scenes = applyDilationToScenes(eroded_scenes, iterations=2, structure=np.ones((3, 3, 3)))  
    del eroded_scenes
    
    # Step 5: Skeletonize and clean, prune skeleton 
    skeletonized_scenes = skeletonizeScenes(dilated_scenes)
    del dilated_scenes
    
    pruned_scenes3D = prune3Dscenes(skeletonized_scenes, size=30) #here removes small branches in 3d as well
    del skeletonized_scenes
    
    z_projected_scenes = zProjectScenes(pruned_scenes3D) # shifting to 2D image 
    del pruned_scenes3D
    
    #cleaned_2d_skeletons = cleanMipSkeleton(z_projected_scenes, min_length=100, max_length=30000) #this 
    
    pruned_scenes, segmented_scenes, segment_objects_list = pruneScenes(cleaned_2d_skeletons, size=30, mask=None) # side branches removal 
    
    skeletonizedscenes = removeLoopsScenes(pruned_scenes)
    del pruned_scenes
    
    # More pruning if needed 
    # pruned_scenes, segmented_scenes, segment_objects_list = pruneScenes(cleaned_2d_skeletons, size=30, mask=None) 

    
    
    
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

def batchProcessFiles(file_list, process_function, folder_path):
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
                "file_index": file_index
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
print(dataframe_results.head)
dataframe_results.to_csv('/Users/tetianasalamovska/Desktop/zeis/df.csv', index=False)

# Example: Get data for file_index 0, scene_index 2
filtered_df = df[(df['file_index'] == 0) & (df['scene_index'] == 2)]

# Example: Get all slices for a specific filename
specific_file_df = df[df['filename'] == 'example_filename.czi']

plot_images(dataframe_results[(dataframe_results['file_name'] == 'IHCT_THT53_40x2x_IHCT08_slice6_stack_positions_A488_laser08_speed6.czi') & (dataframe_results['scene_index'] == 6)], dataframe_results[(dataframe_results['file_name'] == 'IHCT_THT53_40x3x_IHCT08_slice6_stack_positions_A488_laser08_speed6.czi') & (dataframe_results['scene_index'] == 6)], 'Original', 'No Somata')








def release_memory(variable):
    del variable


import matplotlib.pyplot as plt

def plotImages(image1, image2, title1, title2):
    """
    Plot two images side by side for comparison.

    Args:
    image1, image2 (numpy.ndarray): The images to plot.
    title1, title2 (str): Titles for the two images.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image1, cmap='gray')
    axes[0].set_title(title1)
    axes[0].axis('off')  # Turn off axis numbering and ticks

    axes[1].imshow(image2, cmap='gray')
    axes[1].set_title(title2)
    axes[1].axis('off')

    plt.show()



# Print summary of the DataFrame to check its content
print(dataframe_results.head())
print(dataframe_results['file_name'].unique())  # Check the unique filenames to ensure your file is listed
print(dataframe_results['scene_index'].unique())  # Check the unique scene indices
def fetch_and_plot_images(df, file_name_1, scene_index_1, file_name_2, scene_index_2):
    """
    Fetch two images based on file name and scene index, and plot them side by side.
    
    Args:
    df (DataFrame): The DataFrame containing the images and metadata.
    file_name_1 (str): File name of the first image.
    scene_index_1 (int): Scene index of the first image.
    file_name_2 (str): File name of the second image.
    scene_index_2 (int): Scene index of the second image.
    """
    image1_data = df.loc[(df['file_name'] == file_name_1) & (df['scene_index'] == scene_index_1), 'cleaned_scene'].iloc[0]
    image2_data = df.loc[(df['file_name'] == file_name_2) & (df['scene_index'] == scene_index_2), 'cleaned_scene'].iloc[0]
    
    if image1_data is not None and image2_data is not None:
        plot_images(image1_data, image2_data, 'Image 1', 'Image 2')
    else:
        print("One or both of the images could not be found.")

# Usage example
fetch_and_plot_images(dataframe_results, 
                      '/Users/tetianasalamovska/Desktop/zeis/IHCT_THT53_40x2x_IHCT08_slice6_stack_positions_A488_laser08_speed6.czi', 2, 
                      '/Users/tetianasalamovska/Desktop/zeis/IHCT_THT53_40x2x_IHCT08_slice6_stack_positions_A488_laser08_speed6.czi', 9)


# validation and problems in the tg notes 
# check and discuss 
# but overall data is saved normally to the dataframe and is normally executed when 
# needed like above an example with plotting 
# but I maybe need top add file index as well because it can be inconvinient 
# to write whole file name (but you will know for sure with which file are you working with
# --- > decide on the next stage of analysis)



# threshold after TUBENESS

#Step 4.1: Validate tubeness
#validation = subtract_tubeness_from_nosoma(nosoma_scenes[8], tubeness_scenes[8])
#plot_images(tubeness_scenes[8][8,:,:], nosoma_scenes[8][8,:,:], 'Tubeness', 'No Somata')
#plot_images(validation[8,:,:], nosoma_scenes[8][8,:,:], 'Substracted', 'No Somata')
# Inside the function can be changed to see the opposite substraction


