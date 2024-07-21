#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 17:14:19 2024

@author: tetianasalamovska
"""

#!/usr/bin/env python
import sys
import itk
import numpy as np
from distutils.version import StrictVersion as VS
from packaging.version import parse as parse_version


# Check ITK version
if parse_version(itk.Version.GetITKVersion()) < parse_version("5.0.0"):
    print("ITK 5.0.0 or newer is required.")
    sys.exit(1)

# Assume `numpy_image` is your pre-loaded 3D NumPy array 

# Convert NumPy array to ITK image
input_image = itk.GetImageFromArray(adjusted_scenes[1].astype(np.float64))
print(input_image.shape)
# Set parameters directly
sigma_minimum = 1.0
sigma_maximum = 10.0
number_of_sigma_steps = 10

# Define ITK types based on the input image
ImageType = type(input_image)
Dimension = input_image.GetImageDimension()
HessianPixelType = itk.SymmetricSecondRankTensor[itk.F, Dimension]
HessianImageType = itk.Image[HessianPixelType, Dimension]

# Set up the objectness filter
objectness_filter=itk.HessianToObjectnessMeasureImageFilter[HessianImageType, ImageType].New()
objectness_filter.SetBrightObject(False)
objectness_filter.SetScaleObjectnessMeasure(False)
objectness_filter.SetAlpha(0.5)
objectness_filter.SetBeta(1.0)
objectness_filter.SetGamma(5.0)

# Set up the multi-scale Hessian filter
multi_scale_filter = itk.MultiScaleHessianBasedMeasureImageFilter[ImageType, HessianImageType, ImageType].New()
multi_scale_filter.SetInput(input_image)
multi_scale_filter.SetHessianToMeasureFilter(objectness_filter)
multi_scale_filter.SetSigmaStepMethodToLogarithmic()
multi_scale_filter.SetSigmaMinimum(sigma_minimum)
multi_scale_filter.SetSigmaMaximum(sigma_maximum)
multi_scale_filter.SetNumberOfSigmaSteps(number_of_sigma_steps)

# Rescale filter to convert the result to a suitable pixel type
OutputPixelType = itk.UC
OutputImageType = itk.Image[OutputPixelType, Dimension]
rescale_filter = itk.RescaleIntensityImageFilter[ImageType, OutputImageType].New()
rescale_filter.SetInput(multi_scale_filter.GetOutput())

# Convert the output ITK image back to a NumPy array
output_numpy_image = itk.GetArrayFromImage(rescale_filter.GetOutput())

# You can now use `output_numpy_image` inw your Python environment

