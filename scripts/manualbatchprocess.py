#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 14:33:32 2024

@author: tetianasalamovska
"""



# Git clone repository in terminal (instructions in README.txt file)
# Navigate to the directory: this will be replaced with real path on cluster/your computer 

import sys
sys.path.append('/Users/tetianasalamovska/Documents/GitHub/nRIM_curliness_analysis') 


# Importing all related functions after navigating to downloaded repository 
# Also might need requirments.txt file 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from skimage.color import label2rgb
from src.FileImport.DesctiptorsBasedFileSearch import getMatchingFilesList
#from src.FileImport.BatchProcessing import 
from src.FileImport.ReadZeissStacks import readCziFile
from src.ImageProcessing.NormilizeIntensity import normalizeScenes, validateImageAdjustment
from src.FileImport.PlottingImage import plotToCompare, plotImageHistogram, visualize3dMayavi
from src.ImageProcessing.DenoisingFilters import applyGaussian, applyMedianFilter, applyContrastStretching
from src.ImageProcessing.SubstractBackground import subtractBackgroundFromScenes
#from src.ImageProcessing.SatoTubeness import applySatoTubeness 
from src.ImageProcessing.Binarize import removeSomaFromAllScenes, cleanBinaryScenes
from src.ImageProcessing.Skeletonize import skeletonizeScenes, pruneScenes, zProjectScenes, cleanMipSkeleton, prune3Dscenes,removeLoopsScenes, find_branch_pts, breakJunctionsAndLabelScenes  
from src.ImageProcessing.Morphology import applyErosionToScenes, applyDilationToScenes
from src.CurlinessAnalysis.AnalyzeCurliness import analyzeCurliness, visualize_and_analyze_branches
from src.ImageProcessing.Thresholding import otsuThresholdingScenes


# Choosing files you want to analyze, assuming they are all in one folder 
# Define descriptors that you need or just run function without descriptors to get all file names in the folder 

folder_with_data = '/Users/tetianasalamovska/Desktop/zeis'
file_list = getMatchingFilesList(
    folder_with_data,
    EXPERIMENT='IHCT',
    MAGN_LIST=['40x3x'], # you can mention a few ['40x2x', '40x3x'] or leave empty '' for function to ignore it 
    ID='THT53',
    SEPARATOR='_',
    EXTENSION='.czi')

#print("Matching Files:", file_list)


# !!! There must be a batch process function, but I will write a test analysis for 1 file instead 
# and then just put it in the function at src.FileImport.BarchProcessing 

# Step 1. Importing files that match file list names (here just importing one file) and reading metadata
    # 1.1. Read the file as a list of 3D numpy arrays ( or "scenes" like in your single file, there are 11 of them)

file_name = '/Users/tetianasalamovska/Desktop/zeis/IHCT_THT53_40x3x_IHCT08_slice6_stack_positions_A488_laser08_speed6.czi'
scenes, metadata = readCziFile(file_name)

    # 1.2. Create a metadata dictionary and access it with this function 












# Print the shape of scenes just to check 
#if scenes:
    #print(scenes[0-10].shape)


# Step 2. Preprocessing

    # 2.1 Normalizing intensities and enhancing contrast
    
# Also returnes normalized image shape and min and max intensities in the console
# You can print it using print(np.min(scenes[5]))
normalized_scenes = normalizeScenes(scenes, percentiles=[1,99])


# Optionally visualize one of the slices in the one of the stacks before and after 
# Here it is important to remember image shape structure: scenes - your list, [4] - index of scenes,
# in Python index starts from 0, [8,:,:] - this means 8 slice with all X and Y dimension values
plotToCompare(scenes[7][10,:,:], normalized_scenes[7][10,:,:], 'Original', 'Normalized')


validateImageAdjustment(scenes[7], normalized_scenes[7])


# Inspect histograms if needed 
plotImageHistogram(normalized_scenes[6], bins=256, pixel_range=(0, 65535), title='Pixel Intensity Histogram for Normalized Image')

#del scenes

    # 2.2. Denoising and optional morphological techniques
blurred_scenes = applyGaussian(normalized_scenes, sigma=2)

plotToCompare(normalized_scenes[7][10,:,:], blurred_scenes[7][10,:,:], 'Normalized', 'Gaussian Blur')

    # 2.3. Background substraction, try radius from 25 to 35++
    
#for one stack to test different radius 
#subtracted_scene = subtract_background(blurred_scenes[6], radius=35) # 20 is also very fine 

subtracted_scenes = subtractBackgroundFromScenes(blurred_scenes, radius=25)

plotToCompare(subtracted_scenes[7][10,:,:], blurred_scenes[7][10,:,:], 'Substracted', 'Gaussian Blur')

# Might need some contrast enhancement here !!!!!!!!!!!!!!!!


    # Do not use: 2.4. Tubeness filter  (might need more preprocessing before this, more filters or opening/closing operations
    # or pref. demoving some small objects, not noise + contrast enhancement)

# sato tubeness from scikit image 
#tubeness_test = applySatoTubeness(subtracted_scenes[7], sigma = 1, black_ridges=False)
#plotToCompare(subtracted_scenes[7][10,:,:], tubeness_test[10,:,:], 'Substracted', 'Tubeness')


    # 2.5. Contrast enhancement, small obj. cleaning, similar to tubeness!!!!! morphological operation

# opening, closing? clean? stronger filtering here?? other? 

median_scenes = applyMedianFilter(subtracted_scenes, size=4) # choose 3
plotToCompare(subtracted_scenes[7][10,:,:], median_scenes[7][10,:,:], 'Substracted', 'Filter')


stretched_scenes = applyContrastStretching(median_scenes, lower_percentile=1, upper_percentile=99)
plotToCompare(subtracted_scenes[7][10,:,:], stretched_scenes[7][10,:,:], 'Substracted', 'Stretched')
plotToCompare(median_scenes[7][10,:,:], stretched_scenes[7][10,:,:], 'Median scenes', 'Stretched')


# Step 3. Thresholding and binarisation (previously "soma removal") + cleaning
    # 3.1. Thresholding here not adaptive 

#thresholds = 0.4
#nosoma_scenes = removeSomaFromAllScenes(stretched_scenes, thresholds)
#plotToCompare(nosoma_scenes[6][10,:,:], stretched_scenes[6][10,:,:], 'Nosoma', 'Stretched')

binary_scenes = otsuThresholdingScenes(stretched_scenes)
plotToCompare(binary_scenes[7][10,:,:], stretched_scenes[7][10,:,:], 'Binary', 'Stretched')




# ADAPTIVE 
# Instead of using a single threshold value for the whole image, each pixel has its threshold value calculated












    # 3.2. CLeaning 
cleaned_scenes = cleanBinaryScenes(binary_scenes, min_size=4000) #must work in 3D 

plotToCompare(binary_scenes[7][10,:,:], cleaned_scenes[7][10,:,:], 'Binary', 'Cleaned #1')


# Save as tiff or visualize in 3D 
visualize3dMayavi(cleaned_scenes[7])

    # 3.3. erosion, dilation, opening, closing? NEED

# EROSION + DILATION IS THE ANSWER 

eroded_scenes = applyErosionToScenes(cleaned_scenes, iterations=2, structure=np.ones((3, 3, 3)))  # Apply erosion with a 3x3x3 structuring element

plotToCompare(eroded_scenes[6][10,:,:], cleaned_scenes[6][10,:,:], 'eroded scenes', 'Cleaned')

dilated_scenes = applyDilationToScenes(eroded_scenes, iterations=2, structure=np.ones((3, 3, 3)))  # Apply dilation with a 3x3x3 structuring element

plotToCompare(dilated_scenes[7][10,:,:], cleaned_scenes[7][10,:,:], 'after erosion-dilation', 'before erosion-dilation')


visualize3dMayavi(dilated_scenes[7])

###### maybe closing? 






# Step 4. Skeletonization 

    # 4.1. Skeletonization itself 
skeletonized_scenes = skeletonizeScenes(dilated_scenes) # So far so good 


visualize3dMayavi(skeletonized_scenes[7]) # you can save snapshot in this window 


#pruned_img, segmented_img, segment_objects = prune3D(skeletonized_scenes[6], size=30)


#visualize3dMayavi(segmented_img)



    # 4.2. Skeleton pruning and cleaning 
    

# cleaning 




# here cleanskeleotn3d + prune3D functions for scenes  or do Z projection like I am doing here because 

pruned_scenes3D = prune3Dscenes(skeletonized_scenes, size=30)

visualize3dMayavi(pruned_scenes3D[7])



# after erosion and dilation + cleaning, skeleton is simple 

z_projected_scenes = zProjectScenes(pruned_scenes3D)

cleaned_2d_skeletons = cleanMipSkeleton(z_projected_scenes, min_length=20, max_length=30000) #this 
plotToCompare(z_projected_scenes[7], cleaned_2d_skeletons[7], 'MIP', 'clean MIP')


# + pruning 
pruned_scenes, segmented_scenes, segment_objects_list = pruneScenes(cleaned_2d_skeletons, size=30, mask=None)
plotToCompare(pruned_scenes[7], z_projected_scenes[7], 'pruned skeletons', 'MIP')


#skeleton_no_loops = removeLoops(pruned_scenes[6])

#fill holes 
# skeletonize again 
#prune again ? 

final_skeletons = removeLoopsScenes(pruned_scenes)


plotToCompare(pruned_scenes[7], final_skeletons[7], 'pruned skeletons', 'noloops')

skeleton_scenes, segmented_scenes, segment_objects_list = pruneScenes(final_skeletons, size=50, mask=None)
#skeletonized_scenes, segmented_scenes, segment_objects_list = pruneScenes(skeleton_scenes, size=110, mask=None)
# do second round of pruning if needed!!! 



plotToCompare(skeleton_scenes[7], final_skeletons[7], 'cleaned skeletons', 'noloops')






# this can be skipped if dont care about curliness function performance 
# agressive , adjust 
broken_skeletons = breakJunctionsAndLabelScenes(skeleton_scenes, num_iterations=2)

plotToCompare(skeleton_scenes[9], broken_skeletons[9], 'noloops', 'broken skeleton')

from skimage.measure import label 

def measure_connectivity(skeleton, min_size=0):
    """
    Measure the number of connected components in a skeletonized image.
    
    Parameters:
        skeleton (numpy.ndarray): The input skeletonized binary image.
        min_size (int): Minimum size of objects to consider as connected components. Removes smaller artifacts.
        
    Returns:
        int: Number of connected components in the skeletonized image.
    """
    # Convert to uint8 if in another format (True/False, float, etc.)
    skeleton = skeleton.astype(np.uint8)

    # Label connected components
    labeled_skeleton, num_features = label_f(skeleton, connectivity=2, return_num=True)

    # Optionally, remove small components
    if min_size > 0:
        component_sizes = np.bincount(labeled_skeleton.ravel())
        too_small = component_sizes < min_size
        too_small_mask = too_small[labeled_skeleton]
        labeled_skeleton[too_small_mask] = 0
        num_features = np.unique(labeled_skeleton).size - 1  # Exclude background

    return num_features

# Example usage
original_connectivity = measure_connectivity(skeleton_scenes[9])
processed_connectivity = measure_connectivity(broken_skeletons[9])

print(f"Original skeleton connectivity: {original_connectivity}")
print(f"Processed skeleton connectivity: {processed_connectivity}")





# or analyze curliness measures branches not correctly 

#just increase size if you don't want side branches at all 





#################################################### all good up to here but make broke skelet function less harsh
# Analyze Curliness 
# input a df and output extended df with information PER BRANCH (TIDY DATA)
# then completly different analysis for fractality and separate df 
# then compute means from df1 and merge or compare with df2 fractality (per scene level, not per branch)
# to understand it better I need to know how will ny fractality data look like so before putting it into df function 
# I will work on fractality (below this)




curliness, straightness, longest_path_length, max_dendritic_reach, labeled_skeleton, label = analyzeCurliness(broken_skeletons[3])

plotToCompare(broken_skeletons[2], labeled_skeleton, 'cleaned skeletons', 'labeled_skeleton')



visualize_and_analyze_branches(labeled_skeleton, curliness, label, longest_path_length, max_dendritic_reach)




mean_straightness = np.mean(straightness)
mean_curliness = np.mean(curliness)
std_curliness = np.std(curliness)
median_curliness = np.median(curliness)
sem_curliness = std_curliness / np.sqrt(len(longest_path_length))
print(np.median(straightness))



# plot curliness vs length to see the correlation !!!
# longer branches are usually more curly what is expected but not always!!!!


plt.figure()
plt.scatter(longest_path_length, curliness, alpha=0.7)
plt.xlabel('Branch Length')
plt.ylabel('Curliness')
plt.title('Branch Length vs. Curliness')
plt.show()

# check for zero curliness!!!!!!!
for label, curl, length in zip(label, curliness, longest_path_length):
    if curl == 0:
        print(f"Label: {label}, Curliness: {curl}, Length: {length}")


################################## Fractality analysis 

# probably best is working with pruned_scenes or not cleaned z-projected scenes 
# so this is the last step of cleaning 
# + pruning 
#pruned_scenes, segmented_scenes, segment_objects_list = pruneScenes(cleaned_2d_skeletons, size=30, mask=None)
plotToCompare(pruned_scenes[6], z_projected_scenes[6], 'cleaned skeletons', 'MIP')

from src.CurlinessAnalysis.boxCount import boxCount


# Perform box counting
sizes, counts = boxCount(pruned_scenes[0])

# Convert to logarithmic scale
log_sizes = np.log(sizes)
log_counts = np.log(counts)

# Plot the results
plt.figure()
plt.plot(log_sizes, log_counts, 'o-', label='Data')

# Linear fit
coefficients = np.polyfit(log_sizes, log_counts, 1)
fractal_dimension = -coefficients[0]
poly = np.poly1d(coefficients)
plt.plot(log_sizes, poly(log_sizes), 'r--', label=f'Fit (D = {fractal_dimension:.4f})')

plt.xlabel('log(Box Size)')
plt.ylabel('log(Count)')
plt.legend()
plt.title('Box Counting Method')
plt.show()

print(f'Estimated Fractal Dimension: {fractal_dimension:.4f}')

# In the context of a 2D image, the fractal dimension ranges between 1 and 2:
	#•	FD = 1: Represents a simple, linear structure (like a straight line).
	#•	FD = 2: Indicates a structure that completely fills a 2D plane (like a filled square).

# here this is for BATCH DF 

from src.CurlinessAnalysis.boxCount import batchProcessBoxCount

result_frac_df2 = batchProcessBoxCount(file_list, folder_with_data, min_box_size=2)
# comment data that we don't need to include in the df (in the function)























