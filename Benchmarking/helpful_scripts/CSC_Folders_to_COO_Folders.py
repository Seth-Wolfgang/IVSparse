import os
import shutil
import numpy as np
import scipy
import pandas as pd

# Source and destination directories
source_dir = '/home/sethwolfgang/matrices'
destination_dir = '/home/sethwolfgang/matricesCOO'

def csc_to_coo(source_path, destination_path):
    # Load 'inner.csv', 'outer.csv', and 'vals.csv' data as NumPy arrays
    inner_matrix = np.loadtxt(os.path.join(source_path, 'inner.csv'), dtype=int)
    outer_matrix = np.loadtxt(os.path.join(source_path, 'outer.csv'), dtype=int)
    vals_matrix = np.loadtxt(os.path.join(source_path, 'vals.csv'), dtype=float)

    # Create a CSC matrix
    csc_matrix_data = scipy.csc_array(vals_matrix, (inner_matrix, outer_matrix), shape=(max(inner_matrix), len(outer_matrix)))

    # Convert the CSC matrix to COO format
    coo_matrix_data = coo_matrix(csc_matrix_data)

    # Write COO data to separate files
    os.makedirs(destination_path, exist_ok=True)
    np.savetxt(os.path.join(destination_path, 'rows.txt'), coo_matrix_data.row, fmt='%d')
    np.savetxt(os.path.join(destination_path, 'cols.txt'), coo_matrix_data.col, fmt='%d')
    np.savetxt(os.path.join(destination_path, 'vals.txt'), coo_matrix_data.data, fmt='%f')

# Recursively process subdirectories
for root, _, _ in os.walk(source_dir):
    relative_path = os.path.relpath(root, source_dir)
    source_subdir_path = os.path.join(source_dir, relative_path)
    destination_subdir_path = os.path.join(destination_dir, relative_path)

    # Process CSC data if 'inner.csv', 'outer.csv', and 'vals.csv' exist in the source subdirectory
    if os.path.exists(os.path.join(source_subdir_path, 'inner.csv')) and \
       os.path.exists(os.path.join(source_subdir_path, 'outer.csv')) and \
       os.path.exists(os.path.join(source_subdir_path, 'vals.csv')):
        csc_to_coo(source_subdir_path, destination_subdir_path)