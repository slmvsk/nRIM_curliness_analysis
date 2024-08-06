#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 11:59:23 2024

@author: tetianasalamovska
"""

#################################
# function that will find file by name  (there are 2 functions 
# so I need to leave 1, but check if list function works with no list as well)
# IHCT_THT53_40x2x_IHCT08_slice6_stack_positions_A488_laser08_speed6.czi


import glob


def get_matching_files(folder_with_data, **kwargs):
    # Default values for parameters
    default_values = {
        'EXPERIMENT': '',  # Example: 'IHCT'
        'ID': '',          # Example: 'THT53'
        'MAGN': '',   # Example: '40x2x'
         # Slice number??? 
        'SEPARATOR': '-',  # Default separator 
        'EXTENSION': '' #.czi 
    }
    
    # Override default values with provided kwargs
    parameters = {key: kwargs.get(key, default_values[key]) for key in default_values}
    
    # Constructing the base search string
    search_string = parameters['EXPERIMENT']
    
    # Iterate through parameters and construct search_string
    for key, value in parameters.items():
        if key in ['EXPERIMENT', 'SEPARATOR', 'EXTENSION']:
            continue  # Skip these keys as they are not part of the search string
        if value:
            search_string += parameters['SEPARATOR'] + value
        else:
            search_string += '*'
    
    # Print search string for debugging
    print("Search String:", search_string)
    
    # Find files matching the constructed search string
    file_name_list = glob.glob(folder_with_data + '/*' + search_string + '*' + parameters['EXTENSION'])
    
    # Print found files for debugging
    print("Found Files:")
    for file in file_name_list:
        print(file)
    
    return file_name_list


# Example usage
#folder_with_data = '/Users/tetianasalamovska/Desktop/zeis'


# Print all file names to debug
#for file in glob.glob(folder_with_data + '*'):
   # print(file)


# Call the function without any descriptors (will return all files in the folder)
#file_list = get_matching_files(folder_with_data)
#print(file_list)


# Example: Adjust parameters based on your file naming convention
#file_list = get_matching_files(folder_with_data, EXPERIMENT='IHCT', ID='THT53', MAGN='40x3x', SEPARATOR='_')
#print("Matching Files:", file_list)



#####################################################
# function that works with MAGN lists instead 


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
        found_files = glob.glob(folder_with_data + '/*' + search_string + '*' + parameters['EXTENSION'])
        file_name_list.extend(found_files)
    
    # Print found files for debugging
    print("Found Files:")
    for file in file_name_list:
        print(file)
    
    return file_name_list

# Example usage
folder_with_data = '/Users/tetianasalamovska/Desktop/zeis'
#file_list = getMatchingFilesList(
    #folder_with_data,
    #EXPERIMENT='IHCT',
    #MAGN_LIST=['40x2x', '40x3x'],
    #ID='THT53',
    #SEPARATOR='_',
    #EXTENSION='.czi'
#)


