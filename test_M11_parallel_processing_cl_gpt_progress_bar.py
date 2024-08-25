import os
import time
import numpy as np
import psutil
import multiprocessing as mp
from datetime import datetime
from tqdm import tqdm


# do not paralel here 
distance_matrix = np.load('/Users/lakshmipriyaswaminathan/OIST Dropbox/Lakshmipriya Swaminathan/Thesis/Analysis/20240521_parallel_processing_intro/data/M11_20mmin_distance_matrix.npy')
neighbourhood_size_range = np.linspace(0, 1, 100)

def get_avg_nearest_neighbours(args):
    neighbourhood_size, task_id, progress_queue = args
    """A function that takes a given distance matrix and neighbourhood size,
    and returns the average number of nearest neighbours per point."""
    process = psutil.Process(os.getpid())
    nn_m = 0
    num_points, _ = np.shape(distance_matrix)
    for i in range(num_points):
        for j in range(num_points):
            if i != j and distance_matrix[i, j] < neighbourhood_size:
                nn_m += 1
    avg_nn_m = nn_m / (num_points ** 2)

    # Track memory usage
    memory_usage = process.memory_info().rss / (1024 * 1024)  # Memory usage in MB
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    with open(f'task_{task_id}_log_{timestamp}.txt', 'w') as file:
        file.write(f"Task {task_id} (PID: {os.getpid()}) log:\n")
        file.write(f"Memory usage: {memory_usage:.2f} MB\n")

    #print(f'Neighbourhood size {neighbourhood_size} done by task {task_id}!')
    
    # Update progress bar
    progress_queue.put(1)
    
    return avg_nn_m, memory_usage

def monitor_memory(interval=10):
    """Function to periodically monitor memory usage"""
    memory_data = []
    start_time = time.time()
    while True:
        memory_usage = psutil.virtual_memory().used
        memory_data.append((time.time(), memory_usage))
        time.sleep(interval)
        if time.time() - start_time > 10:  # Monitor for 10 seconds
            break
    return memory_data

def worker_wrapper(args):
    """Wrapper for worker function to handle progress bar updates"""
    result = get_avg_nearest_neighbours(args)
    return result

if __name__ == "__main__":
    start_time = time.time()
    calculations = len(neighbourhood_size_range)
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_name = f"output_log_{calculations}_{current_datetime}.txt"
    log_file_path = f"/Users/lakshmipriyaswaminathan/OIST Dropbox/Lakshmipriya Swaminathan/Thesis/Analysis/20240521_parallel_processing_intro/logs/{log_file_name}"

    # Start memory monitoring
    memory_data = monitor_memory()

    # Prepare arguments for workers
    manager = mp.Manager()
    progress_queue = manager.Queue()
    task_args = [(neighbourhood_size, idx, progress_queue) for idx, neighbourhood_size in enumerate(neighbourhood_size_range)]

    # Create and start progress bar
    results = []
    with tqdm(total=len(task_args)) as pbar:
        with mp.Pool(processes=5) as pool:
            # Map worker function with progress queue
            for result in pool.imap_unordered(worker_wrapper, task_args):
                results.append(result)
                pbar.update(1)

    # Separate results and memory usage
    avg_nearest_neighbours, memory_usages = zip(*results)

    print('Correlation integral computed')
    elapsed_time = time.time() - start_time
    print('Time Elapsed (Parallel): ', elapsed_time)

    # Save results as a numpy array
    results_array = np.array(avg_nearest_neighbours)
    np.save('/Users/lakshmipriyaswaminathan/OIST Dropbox/Lakshmipriya Swaminathan/Thesis/Analysis/20240521_parallel_processing_intro/data/M11_cl_10000steps_pp_progress_bar_test.npy', results_array)
    print('Saved results')

    # Write all data to the output log file
    with open(log_file_path, "w") as f:
        f.write("Time (s), Memory Usage (MB)\n")
        for time_point, memory_usage in memory_data:
            f.write(f"{time_point:.2f}, {memory_usage:.2f}\n")
        f.write(f"Total Time (s): {elapsed_time:.2f}\n")
        f.write(f"Total Memory Usage (MB): {psutil.virtual_memory().used / (1024 * 1024):.2f}\n")

        # Add memory usage for each task
        f.write("\nMemory Usage per Task (MB):\n")
        for idx, mem_usage in enumerate(memory_usages):
            f.write(f"Task {idx}: {mem_usage:.2f} MB\n")
