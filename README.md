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
git clone https://github.com/yourusername/dendrite-analysis-pipeline.git
cd dendrite-analysis-pipeline


## Usage


How to use the project.

## Contributing

Guidelines for contributing to the project.

## License

Details about the license.

## Contact

How to reach the project's maintainer.
