#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:15:33 2024

@author: lakshmipriyaswaminathan
"""

#for correlation dimension
import numpy as np
import multiprocessing
import time
import os
from tqdm import tqdm
import sys
import umap
import joblib
sys.path.append(r'/Users/lakshmipriyaswaminathan/Documents/GitHub/Scratchy/')
# make sure you added scratchy to the path!
import original_code.scratchy as sy # all my functions are here


#writing a section that computer the delay matrix and saves that and the sample names
data_path = '/Users/lakshmipriyaswaminathan/OIST Dropbox/Lakshmipriya Swaminathan/Thesis/figures/20240725_distance_umap_20mmin/M11/mocap_data/TTM2_S23_M11_MCL7_T2_TREADMILL2_T01_26102021_aimmodelprocessed_lps_01122021_lgr_5062022.mat'
#data_name = 
#mat_data = sy.loadmat(data_path)
print('ct of ' +data_path)
ct_output, ct_output_mean, ct_timepoints = sy.wrapper_ct(data_path)

print('building trajectory matrices')
master_delay_matrix_list = []
#sample_name_list = []
for i in range(1,11):
    print('delay :  '+str(i)+' frames')
    tm = sy.trajectory_matrix_nd(ct_output, i)
    savename = 'delay_embed_M11_'+str(i).zfill(6)+'_frames.npy'
    np.save('/Users/lakshmipriyaswaminathan/OIST Dropbox/Lakshmipriya Swaminathan/Thesis/figures/20240725_distance_umap_20mmin/M11/delay_matrix_data/'+savename,tm)
    master_delay_matrix_list.append(tm)
    #sample_name_list.append('delay_'+str(i).zfill(6)+'_frames')
    
#now we do the UMAP embeddings
#umap_embedding_list = []
print('making and saving UMAP model')
embedding_3d = umap.UMAP(n_neighbors=16, n_components=3)
joblib_file = '/Users/lakshmipriyaswaminathan/OIST Dropbox/Lakshmipriya Swaminathan/Thesis/figures/20240725_distance_umap_20mmin/M11/UMAP_data/umap_embeddings/umap_object_M11_20240725.pkl'
joblib.dump(embedding_3d, joblib_file)

print('fitting delay embeddings to 3d UMAP')
for i in range(len(master_delay_matrix_list)):
    print('delay :  '+str(i)+' frames')
    tm = master_delay_matrix_list[i]
    u_embed = embedding_3d.fit_transform(tm.T)
    savename = 'umap_delay_'+str(i).zfill(6)+'_frames.npy'
    np.save('/Users/lakshmipriyaswaminathan/OIST Dropbox/Lakshmipriya Swaminathan/Thesis/figures/20240725_distance_umap_20mmin/M11/UMAP_data/umap_embeddings/'+savename, u_embed, allow_pickle=True)
    #umap_embedding_list.append(u_embed)
    
    

# distance part
def compute_distance_chunk(timeseries, norm, start_idx, end_idx):
    size = timeseries.shape[1]
    chunk_matrix = np.zeros((end_idx - start_idx, size))
    
    for i in range(start_idx, end_idx):
        for j in range(size):
            if norm == 'euclidean':
                dist = np.linalg.norm(timeseries[:, i] - timeseries[:, j])
            elif norm == 'manhattan':
                dist = np.sum(np.abs(timeseries[:, i] - timeseries[:, j]))
            else:
                raise ValueError("Unsupported norm. Use 'euclidean' or 'manhattan'.")
            chunk_matrix[i - start_idx, j] = dist
    
    return chunk_matrix
    pass

def distance_matrix_parallel(timeseries, norm='euclidean', num_cores=4):
    if np.ndim(timeseries) > 1:
        a, b = np.shape(timeseries)
        if a > b:
            size = a
            timeseries = np.transpose(timeseries)
        else:
            size = b
    elif np.ndim(timeseries) == 0:
        raise ValueError("I need a non-zero dimensional timeseries to proceed!")
    else:
        size = len(timeseries)

    chunk_size = size // num_cores
    indices = [(i, min(i + chunk_size, size)) for i in range(0, size, chunk_size)]

    start_time = time.time()  # Start time logging
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.starmap(compute_distance_chunk, [(timeseries, norm, start, end) for start, end in indices]),
                            total=len(indices), desc="Computing distance matrix"))

    end_time = time.time()  # End time logging

    distance_matrix = np.vstack(results)
    max_dist = np.max(distance_matrix)
    distance_matrix = distance_matrix / max_dist

    # Set the diagonal to infinity to ignore zero distances when finding the minimum
    np.fill_diagonal(distance_matrix, np.inf)
    min_dist = np.min(distance_matrix)
    min_dist = min_dist / max_dist  # Normalize the minimum distance

    computation_time = end_time - start_time

    return distance_matrix, max_dist, min_dist, computation_time
    pass

def save_results(distance_matrix, max_dist, min_dist, computation_time, sample_name, output_dir):
    # Save the distance matrix as a numpy array
    distance_matrix_file = os.path.join(output_dir, f"{sample_name}_distance_matrix.npy")
    np.save(distance_matrix_file, distance_matrix)
    
    # Create the log content
    log_content = (f"Sample: {sample_name}\n"
                   f"Max Distance: {max_dist}\n"
                   f"Min Normalized Distance: {min_dist}\n"
                   f"Computation Time: {computation_time} seconds\n"
                   f"Distance Matrix File: {distance_matrix_file}\n")
    
    # Save the log as a text file
    log_file = os.path.join(output_dir, f"{sample_name}_log.txt")
    with open(log_file, 'w') as f:
        f.write(log_content)
    
    return distance_matrix_file, log_file
    pass

def process_samples(samples, sample_names, output_dir, norm='euclidean', num_cores=4):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for sample, sample_name in zip(samples, sample_names):
        print(f"Processing sample: {sample_name}")
        distance_matrix, max_dist, min_dist, computation_time = distance_matrix_parallel(sample, norm=norm, num_cores=num_cores)
        save_results(distance_matrix, max_dist, min_dist, computation_time, sample_name, output_dir)
    pass

# Example usage

# samples = [np.random.rand(8, 10000) for _ in range(3)]  # Replace with your actual samples
# sample_names = ['sample1', 'sample2', 'sample3']  # Replace with your actual sample names
# output_dir = './output'
# process_samples(samples, sample_names, output_dir, norm='euclidean', num_cores=4)




#load samples

if __name__ == '__main__':
    samples = []
    data_dir = '/Users/lakshmipriyaswaminathan/OIST Dropbox/Lakshmipriya Swaminathan/Thesis/figures/20240725_distance_umap_20mmin/M11/UMAP_data/umap_embeddings/'
    fnames_ = os.listdir(data_dir)
    fnames = [s for s in fnames_ if s.startswith('u')]
    #os.chdir(data_dir)
    print('loading files from' + data_dir)
    samples = [np.load(data_dir+'/'+f, allow_pickle=True) for f in fnames ]
    #samples = [s[:8,:] for s in samples_]
    print('loaded files: '+str(len(samples)))
    #now that it's loaded
    #let's get sample names
    print('getting sample names')
    sample_names = [s.replace('.npy','') for s in fnames]

    output_dir = '/Users/lakshmipriyaswaminathan/OIST Dropbox/Lakshmipriya Swaminathan/Thesis/figures/20240725_distance_umap_20mmin/M11/distance_matrices/'
    print('setting output directory to: '+output_dir)
    print('starting distance_matrix calculations')
    process_samples(samples, sample_names,output_dir,norm='euclidean',num_cores=4)
    print('process finished and files saved to '+output_dir)
