#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 12:55:55 2024

@author: tetianasalamovska
"""

def box_count(binary_image, min_box_size=2):
    import numpy as np
    
    """ Calculate Box Sizes: Use powers of 2 to define box sizes.
    	Loop Over Box Sizes: For each box size, divide the image into boxes.
        Count Non-Empty Boxes: For each box, check if it contains any part of the dendrite.
        Collect Counts: Store the number of non-empty boxes for each box size. """

    # Get the size of the image
    rows, cols = binary_image.shape

    # Maximum box size is the size of the image
    n = min(rows, cols)
    sizes = 2 ** np.arange(np.floor(np.log2(n)), np.floor(np.log2(min_box_size)) - 1, -1).astype(int)

    counts = []
    for size in sizes:
        # Number of boxes along each dimension
        n_rows = int(np.ceil(rows / size))
        n_cols = int(np.ceil(cols / size))

        # Count the number of non-empty boxes
        count = 0
        for i in range(n_rows):
            for j in range(n_cols):
                # Define the box boundaries
                row_start = i * size
                row_end = min((i + 1) * size, rows)
                col_start = j * size
                col_end = min((j + 1) * size, cols)

                # Extract the box
                box = binary_image[row_start:row_end, col_start:col_end]

                # Check if the box contains any foreground pixels
                if np.any(box):
                    count += 1

        counts.append(count)

    return sizes, counts


# Example usage 


# Perform box counting
#sizes, counts = box_count(pruned_scenes[7])

# Convert to logarithmic scale
#log_sizes = np.log(sizes)
#log_counts = np.log(counts)

# Plot the results
#plt.figure()
#plt.plot(log_sizes, log_counts, 'o-', label='Data')

# Linear fit
#coefficients = np.polyfit(log_sizes, log_counts, 1)
#fractal_dimension = -coefficients[0]
#poly = np.poly1d(coefficients)
#plt.plot(log_sizes, poly(log_sizes), 'r--', label=f'Fit (D = {fractal_dimension:.4f})')

#plt.xlabel('log(Box Size)')
#plt.ylabel('log(Count)')
#plt.legend()
#plt.title('Box Counting Method')
#plt.show()

#print(f'Estimated Fractal Dimension: {fractal_dimension:.4f}')