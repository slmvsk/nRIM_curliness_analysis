#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:46:35 2024

Enhance features 

@author: tetianasalamovska
"""




# Mask function insert 

 def enhance_neurites(self, image, radius, method):
     data = self.__mask(image.pixel_data, image.mask)
     
         sigma = self.smoothing.value

         smoothed = scipy.ndimage.gaussian_filter(
             data, numpy.divide(sigma, image.spacing)
         )

         if image.volumetric:
             result = numpy.zeros_like(smoothed)

             for index, plane in enumerate(smoothed):
                 hessian = centrosome.filter.hessian(
                     plane, return_hessian=False, return_eigenvectors=False
                 )

                 result[index] = (
                     -hessian[:, :, 0] * (hessian[:, :, 0] < 0) * (sigma ** 2)
                 )
         else:
             hessian = centrosome.filter.hessian(
                 smoothed, return_hessian=False, return_eigenvectors=False
             )

             #
             # The positive values are darker pixels with lighter
             # neighbors. The original ImageJ code scales the result
             # by sigma squared - I have a feeling this might be
             # a first-order correction for e**(-2*sigma), possibly
             # because the hessian is taken from one pixel away
             # and the gradient is less as sigma gets larger.
             result = -hessian[:, :, 0] * (hessian[:, :, 0] < 0) * (sigma ** 2)

     return self.__unmask(result, image.pixel_data, image.mask)