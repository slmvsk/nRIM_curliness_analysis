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

###  Step 1: Read the file as a list of 3D numpy arrays (scenes)
This functionreads a ZEISS CZI file and returns a list of 3D numpy arrays (one per scene) and metadata. It extracts scenes assuming shape (1, 1, S, 1, 1, Z, Y, X, 1), where S is number of scenes (scene index in the extracted form). Data has 3 dimensions (X, Y, Z) and is a single channel, that's why other 1D's doesn't matter and in the end we will have a list of scenes (with indexes starting at 0) with the shape (Z, X, Y) for example (20, 1024, 1024). Understanding of the numpy arrays and data shape will help us access specific slices in these lists. For example if I want to chech middle slice of third image in my list, I will plot these: scenes[2],[10, :,:], where ":" will mean to include all pixels. 

```
scenes, metadata = readCziFile(file_name)
```
To access metadata you will need ...


### Step 2: Normalize intensity 

To normalize intensity we will apply the linear contrast stretching to an entire 3D image stack globally with adjustable percentiles that are parameters to adjust in this function. The linearContrastStretching function enhances the contrast of a 3D image stack by applying linear contrast stretching based on adjustable percentiles. It rescales the pixel intensities of each slice within the stack to a defined output range, making details more visible. It only accepts np.uint8 or np.uint16 data types. The percentiles parameter allows control over which intensity values are stretched, making this function adaptable to different image characteristics, they define the lower and upper thresholds for intensity scaling. 

```
normalized_scenes = normalizeScenes(scenes, percentiles=[1,99])
```
Example of normalizing intensities for one slice of one scene (=image stack): 

<img width="600" alt="Screenshot 2024-09-29 at 14 31 59" src="https://github.com/user-attachments/assets/27d54dab-4e73-4cf4-b73b-548ecf7c5226">


You can also apply validation function validateImageAdjustment and check the output: 

Scene shape: (20, 1024, 1024)
Scene - min, max: 4 65535
Adjusted scene shape: (20, 1024, 1024)
Adjusted scene - min, max: 0 65535


### Step 3: Denoising 

Dendrites in these images will be impossible to segment without some of denoising steps, morphological orerations etc. First thing will be common denoising techniques like Gaussian blur, which will help with sand noise and will smooth edges a little bit. 

```
blurred_scenes = applyGaussian(normalized_scenes, sigma=2)
```

It has sigma parameter that represents the standard deviation of the Gaussian distribution used to create the blur. Higher sigma = more significant smoothing effect, because it means that when we apply a kernel, pixels farther from the central pixel contribute more to the final blurred pixel value, leading to a more pronounced blur effect over a wider area. 

<img width="584" alt="Screenshot 2024-09-29 at 14 37 58" src="https://github.com/user-attachments/assets/9196cdbe-01e9-4e3d-976c-6ace74f4080b">


### Step 4: Background substraction and contrast enhancement 

Next step will be the substract background (with all its noise) function that uses a white top-hat filter on a 3D image stack. The function creates a 2D disk-shaped structuring element with a specified radius. This element helps identify and subtract the background from each slice of the 3D stack.

```
subtracted_scenes = subtractBackgroundFromScenes(blurred_scenes, radius=25) # 25 is fine
```
A larger radius means the filter will subtract broader and larger background structures, useful for images with significant background noise or large objects.
A smaller radius focuses on finer details, removing smaller background elements while retaining the primary features.

<img width="584" alt="Screenshot 2024-09-29 at 14 49 14" src="https://github.com/user-attachments/assets/9c7d065f-2d86-4ab4-b74c-dc14eeb312b3">

After that I am applying median filter that is a brother-filter of Gaussian blur. The size defines the dimensions of the filtering window (kernel) used to process the image. This window slides across the image, replacing each pixel’s value with the median value of all the pixel values within the window. For 2D it requires 2 dimensions and for 3D - 3, but when one single integer is provided, the filter will use a square window (ex. 3 x 3 x 3). Smaller size preserves more details and edges in the image.

```
median_scenes = applyMedianFilter(subtracted_scenes, size=3)
```
This step is not mandatory and by eye you will not see a difference. 

Next step will be applying contrast stretching again, which is optional, but recommended for better visualisation. 

```
stretched_scenes = applyContrastStretching(median_scenes, lower_percentile=1, upper_percentile=99)
```

<img width="582" alt="Screenshot 2024-09-29 at 14 57 44" src="https://github.com/user-attachments/assets/11d02ac8-02b0-4709-8b3b-7d3ccac313a9">


### Step 5: Remove soma / Thresholding / Binarising / Segmenting 

Here the thresholding take place. There was removeSoma function from Binarise.py that finds optimal thresholds, but for now I apply Otsu thresholding (from src.ImageProcessing.Thresholding) to binarise images (now we work with zeros and ones, where 0 is background). Ideally, you apply adaptive thresholding and there are commented functions for that in source code, but they need to be tested and validated (+fixed if needed). If making code/pipeline better - this is the place to start. 

```
binary_scenes = otsuThresholdingScenes(stretched_scenes) # maybe put old nosoma adaptive thresholding function back here ? 
```

<img width="587" alt="Screenshot 2024-09-29 at 15 05 33" src="https://github.com/user-attachments/assets/2fa0fab6-2145-46cf-862b-76b91e5099e8">

### Step 6: Cleaning

If not applying fancy thresholding, then you need to clean you images after binarising them better. 
First step is removing small objects (objects smaller that min_size). This function works with 3D connectivity so It takes in to the count Z-dimension. The min_size is measured in pixels (voxels for 3D images). 

```
cleaned_scenes = cleanBinaryScenes(binary_scenes, min_size=4000)
```
<img width="583" alt="Screenshot 2024-09-29 at 15 14 21" src="https://github.com/user-attachments/assets/c46b62e3-ed9a-4025-b30c-5a51c5278b8e">

You can still see small pixels because some of them are large enough in Z-dimension. This is a 3D view from X axis. 

<img width="443" alt="Screenshot 2024-09-29 at 15 16 45" src="https://github.com/user-attachments/assets/4d1fcaa1-b1d8-446f-92b3-0584c63e5233">

Then I apply classic erosion-dilation steps to smooth image a little bit more and remove not useful "spines".  The principle of the parameters here are similar to what we alredy saw before. In morphological operations like erosion, a structuring element determines the shape and size of the neighborhood around each pixel or voxel (for 3D images). The structuring element acts as a “template” that slides over the image to apply the morphological transformation. Erosion will examine each voxel in the 3D image and compare it with its neighbors defined by the structuring element. In this case, the neighbors form a 3x3x3 cube around each voxel.If any part of this cube (structuring element) contains a 0 when centered on the voxel, that voxel will be set to 0 in the output image. Thus, erosion tends to “shrink” objects, especially at their edges, by removing pixels/voxels.

Dilation will examine each voxel in the 3D image and compare it with its neighbors defined by the structuring element. In this case, the neighbors form a 3x3x3 cube around each voxel. If any part of this cube (structuring element) overlaps with a “1” (foreground pixel/voxel) when centered on the voxel, that voxel will be set to 1 in the output image. Therefore, dilation tends to “expand” objects, adding pixels/voxels around their edges.

```
eroded_scenes = applyErosionToScenes(cleaned_scenes, iterations=2, structure=np.ones((3, 3, 3)))

dilated_scenes = applyDilationToScenes(eroded_scenes, iterations=2, structure=np.ones((3, 3, 3)))  
```

<img width="589" alt="Screenshot 2024-09-29 at 15 23 17" src="https://github.com/user-attachments/assets/029d8f86-a901-4543-bf6a-1ac6922ef0f2">

The difference is small, but It helps when working with binary image and you are limited only to morphological operations. 

### Step 7: Skeletonize and clean, prune skeleton 

For curliness analysis we need to skeletonize our images, Z-rpject them (from here we work with 2D), prune (remove small side branches) and remove small objects. 
The prune2D function (for batch - pruneScenes)  removes small branches from a skeletonized 2D image based on a specified length (size). It identifies individual skeleton segments, retains those longer than size, and prunes shorter ones. The function returns the cleaned (pruned) skeleton image, a segmented version of the pruned skeleton, and a list of the remaining segment contours. This is the main operation here that will help us untangle skeletons when projected. 

```
skeletonized_scenes = skeletonizeScenes(dilated_scenes)

```

Skeletonized image example (3D): 

<img width="489" alt="Screenshot 2024-09-29 at 15 33 03" src="https://github.com/user-attachments/assets/8f937290-24c8-486b-84ea-3fe667a5d560">

```
pruned_scenes3D = prune3Dscenes(skeletonized_scenes, size=30) #here removes small branches in 3d as well
```
After "pruning" in 3D: 

<img width="554" alt="Screenshot 2024-09-29 at 15 34 25" src="https://github.com/user-attachments/assets/bfcd8040-5022-4271-acd5-d6cbe02b6965">

```
z_projected_scenes = zProjectScenes(pruned_scenes3D) # shifting to 2D image 

cleaned_2d_skeletons = cleanMipSkeleton(z_projected_scenes, min_length=100, max_length=30000) #this
```

Z-projected vs small objects removed (here you can adjust size and remove more):
<img width="631" alt="Screenshot 2024-09-29 at 15 36 15" src="https://github.com/user-attachments/assets/e64fad59-43da-430c-8ffc-ecfdef37a44f">

```
pruned_scenes, segmented_scenes, segment_objects_list = pruneScenes(cleaned_2d_skeletons, size=30, mask=None) # side branches removal

```
<img width="631" alt="Screenshot 2024-09-29 at 15 39 36" src="https://github.com/user-attachments/assets/e3a352d6-25e3-4427-ba9c-00fff4ef05ac">

Of course, you can do it more aggresive and you will not need next steps, but I want to save as much as possible of small objects, because they still have curliness information. 


### Step 8: Removing loops and breaking junctions 

Thses functions just do what they say. They are crucial for curliness analysis. 
```
skeletonizedscenes = removeLoopsScenes(pruned_scenes)
```
Result: 

<img width="632" alt="Screenshot 2024-09-29 at 15 42 22" src="https://github.com/user-attachments/assets/4bcaffd6-4534-48d9-b38f-9c5e9331f73c">

```
output_skeletons = breakJunctionsAndLabelScenes(skeletonizedscenes, num_iterations=3)
``` 
Result: 




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
