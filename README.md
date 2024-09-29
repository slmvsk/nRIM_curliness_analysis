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


## Usage

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

If performed on the skeletonized dendrites, it shows lower "fractality" in the range of 1.2-1.4, and when performed on original-wide binary image, it is showing FD in the range of 1.6 - 1.8, which are more realistic numbers for neuropil to give. You can easily adjust the input images by commenting or uncommenting some preprocessing steps in the main preprocessing function processFileforFrac (src.FileImport.BatchFunction). 






How to use the project.

## Contributing

Guidelines for contributing to the project.

## License

Details about the license.

## Contact

How to reach the project's maintainer.
