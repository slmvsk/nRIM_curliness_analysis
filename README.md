# Automated 3D Image Analysis Pipeline for Dendritic Morphology Characterisation in Olivary Neuropil 

In this repository you can fins automated pipeline for confocal image analysis. 
The pipeline processes CZI files, extracts metadata, segments dendrites, removes somatas from image, skeletonizes the images, and cleans the resulting skeletons for further morphological analysis, 
including fractal dimension calculations and dendrite curliness analysis.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)

## Introduction

This project is primarily focused on automating the analysis of 3D images, specifically neuron dendrite structures, using a variety of image processing techniques. 
The workflow handles large datasets from Zeiss CZI files, processes each scene as a 3D numpy array, and includes custom functions for neuron image analysis.

Example images: 
![image](https://github.com/user-attachments/assets/1e60087e-0b71-4684-82eb-b6ded36cc806)



## Installation

Make sure to install the following libraries before running the project:
-	Python 3.8+
- NumPy: For handling multi-dimensional arrays.
-	Pandas: For organizing and managing the output data.
-	czifile: For reading Zeiss CZI files.
-	scikit-image: For image processing functions like binarization and skeletonization.
-	OpenCV: For additional image processing functions (if needed).
-	Matplotlib: For visualizing results.
-	scikit-learn: For advanced image cleaning and analysis functions.

To install all the required packages, you can run this in your computer's terminal:

```bash
pip install numpy pandas czifile scikit-image opencv-python matplotlib scikit-learn
```
After this, you need to clone this repository to your local machine (basically download it):

```bash
git clone https://github.com/slmvsk/nRIM_curliness_analysis.git
cd nRIM_curliness_analysis
```
Install the dependencies listed in the requirements.txt file:

```bash
pip install -r requirements.txt
```

Now ensure your image files (CZI format) are in the correct single folder structured as expected by the script. File names must be written like ...

### Folder structure

There are two main folders: src and scripts. In the src folder you can find files with functions that are sorted in 3 main categories: FileImport (reading images, metadata extraction, plotting and batch processing), CurlinessAnalysis (also has fractality analysis), and ImageProcessing (everything from normalizing images to skeletonization). 

You will mainly work with scripts, unless you will need to adjust some parameters in one of the cases. There are 2 scripts: one for manual run with plotting and each step validation with the opportunity to analyze one or few files ajusting parameters in real time. It is called "manualbatchprocess.py". Another script "runAnalysis" is made for you to batch process files without your supervision. You just insert a folder of your interest and collect output dataframes later. Let's start with the second case when you acctualy need to look into source code to adjust settings in the src function ProcessFile (src.FileImport.BatchFunction) before running it in the headless mode. 

I recommend test your analysis with first manual script, playing around with parameters and find the best optimal compromise in the preprocessing pipeline. Then go to src function ProcessFile (src.FileImport.BatchFunction) that consists of all the steps in the manual script but without plotting, and comment or uncomment some steps or adjust parameters if needed the same way you have it in manual script. Then import or run this function again, set your folder_with_data path and press "Run" button. The first script is more for analysis under someone's control, maybe you want to control the process in the details and analyse files one by one plotting each "scene" (image) after each step or checking images shapes or intensity distributions etc. 

## Preprocessing steps 

My main processing function includes steps: 

```
# Step 1: Read the file as a list of 3D numpy arrays (scenes)
scenes, metadata = readCziFile(file_name)
```
```
# Step 2: Normalize intensity 
normalized_scenes = normalizeScenes(scenes, percentiles=[1,99])
```
```
# Step 3: Denoising 
blurred_scenes = applyGaussian(normalized_scenes, sigma=2)
```

```
# Step 4: Background substraction and contrast enhancement 
subtracted_scenes = subtractBackgroundFromScenes(blurred_scenes, radius=25)

median_scenes = applyMedianFilter(subtracted_scenes, size=3)

stretched_scenes = applyContrastStretching(median_scenes, lower_percentile=1, upper_percentile=99)
```

    
```
# Step 5: Remove soma
binary_scenes = otsuThresholdingScenes(stretched_scenes) # maybe put old nosoma adaptive thresholding function back here 
```
   
```
# Step 6: Cleaning
cleaned_scenes = cleanBinaryScenes(binary_scenes, min_size=4000) 

eroded_scenes = applyErosionToScenes(cleaned_scenes, iterations=2, structure=np.ones((3, 3, 3)))

dilated_scenes = applyDilationToScenes(eroded_scenes, iterations=2, structure=np.ones((3, 3, 3)))  
```

    
```
# Step 7: Skeletonize and clean, prune skeleton 
skeletonized_scenes = skeletonizeScenes(dilated_scenes)

pruned_scenes3D = prune3Dscenes(skeletonized_scenes, size=30) #here removes small branches in 3d as well

z_projected_scenes = zProjectScenes(pruned_scenes3D) # shifting to 2D image 

cleaned_2d_skeletons = cleanMipSkeleton(z_projected_scenes, min_length=100, max_length=30000) #this 
    
pruned_scenes, segmented_scenes, segment_objects_list = pruneScenes(cleaned_2d_skeletons, size=30, mask=None) # side branches removal 
```

```
# Step 8: Removing loops and breaking junctions 
skeletonizedscenes = removeLoopsScenes(pruned_scenes)

output_skeletons = breakJunctionsAndLabelScenes(skeletonizedscenes, num_iterations=3)
``` 



## Analysis 

### Curliness Analysis 

Curliness is a measure that quantifies how much a branch deviates from being straight. In this analysis, we compute the curliness of each branch in a skeletonized image by comparing the Euclidean distance between the endpoints of the branch to the actual path length along the branch (geodesic distance). 

The input must be skeletonized image optionally after pruning (removing small side branches), but breaking junction action and removing loops action must be done to accurately calculate curliness. 

To compute curliness, the function firstly computes straightness, which is defined as the ratio of the Euclidean distance to the geodesic path length.

$$
\text{Straightness} = \text{Euclidean Distance}/\text{Geodesic Path Length} 
$$

Straightness values range from 0 to 1, where 1 indicates a perfectly straight branch. 
Curliness is defines as: 

$$
\text{Curliness} = 1 - \text{Straightness}
$$

Curliness values range from 0 to 1, where 1 indicates maximum curliness. 




### Fractality Analysis 

Here I also implement box-counting method to estimate the fractal dimension of 2D images of dendritic structures within the neuropil. It works by overlaying a grid of boxes of varying sizes onto the image and counting the number of boxes that contain part of the structure. By analyzing how this count scales with the size of the boxes, we can estimate the fractal dimension. Values typically range between 1 and 2 for 2D images. Higher values indicate more complex and space-filling dendritic structures.

The output of the core boxCount function is log-log plot of box sizes (the sizes of the boxes used in the box-counting method, typically powers of 2) and box counts (the number of boxes at each size that contain part of the structure) with linear fit to the data. Each point on the plot represents the log-transformed count for a specific box size. A straight-line pattern on the log-log plot suggests a fractal (self-similar) structure. Deviations from linearity may indicate non-fractal behavior or scales where the fractal approximation is less accurate. Then the linear fit provides the slope and intercept of the best-fit line through the data points. The slope of the line is directly related to the fractal dimension, because fractal dimension (FD) is calculated as: 

$$
\text{Fractal Dimension (FD)} = -\text{slope}
$$

where the negative sign accounts for the inverse relationship between box size and count. 


<img width="417" alt="Screenshot 2024-09-29 at 13 40 43" src="https://github.com/user-attachments/assets/3d61e8e7-ca64-4912-9bb4-91e9b010ae31">



If performed on the skeletonized dendrites, it shows lower "fractality" in the range of 1.2-1.4, and when performed on original-wide binary image, it is showing FD in the range of 1.6 - 1.8, which are more realistic numbers for neuropil to give. You can easily adjust the input images by commenting or uncommenting some preprocessing steps in the main preprocessing function processFileforFrac (src.FileImport.BatchFunction). 








How to use the project.

## Contributing

Guidelines for contributing to the project.

## License

Details about the license.

## Contact

How to reach the project's maintainer.
