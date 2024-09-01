#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 11:59:23 2024

@author: tetianasalamovska
"""


# Function that will find file by name descriptors
# Example name of the file 
# IHCT_THT53_40x2x_IHCT08_slice6_stack_positions_A488_laser08_speed6.czi

import os
import glob

def getMatchingFilesList(folder_with_data, **kwargs):
    # Default values for parameters
    default_values = {
        'EXPERIMENT': '',  # Example: 'IHCT'
        'ID': '',          # Example: 'THT53'
        'MAGN_LIST': [],   # Example: ['40x2x', '40x3x']
        'SEPARATOR': '-',  # Default separator
        'EXTENSION': '.czi' # Default extension
    }
    
    # Override default values with provided kwargs
    parameters = {key: kwargs.get(key, default_values[key]) for key in default_values}
    
    # Ensure MAGN_LIST is a list
    if isinstance(parameters['MAGN_LIST'], str):
        parameters['MAGN_LIST'] = [parameters['MAGN_LIST']]
    
    # If no magnification list provided, default to all files
    if not parameters['MAGN_LIST']:
        search_string = '*' + parameters['EXTENSION']
        found_files = glob.glob(os.path.join(folder_with_data, search_string))
        print("No specific magnification provided. Searching all files.")
    else:
        # Find files matching the constructed search string for each magnification
        file_name_list = []
        for magn in parameters['MAGN_LIST']:
            search_string = parameters['EXPERIMENT']
            
            if parameters['ID']:
                search_string += parameters['SEPARATOR'] + parameters['ID']
            
            search_string += parameters['SEPARATOR'] + magn
            
            # Print search string for debugging
            print("Search String:", search_string)
            
            # Find files matching the constructed search string
            found_files = glob.glob(os.path.join(folder_with_data, '*' + search_string + '*' + parameters['EXTENSION']))
            file_name_list.extend(found_files)
    
        # Print found files for debugging
        print("Found Files:")
        for file in file_name_list:
            print(file)
        
        return file_name_list

# Print all file names to debug
#for file in glob.glob(folder_with_data + '*'):
   # print(file)
