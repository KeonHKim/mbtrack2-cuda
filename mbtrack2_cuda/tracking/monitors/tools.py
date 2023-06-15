# -*- coding: utf-8 -*-
"""
This module defines utilities functions, helping to deals with tracking output 
and hdf5 files.
"""

import numpy as np
import h5py as hp

def merge_files(files_prefix, files_number, start_idx=0, file_name=None):
    """
    Merge several hdf5 files into one.
    
    The function assumes that the files to merge have names in the follwing 
    format:
        - "files_prefix_0.hdf5"
        - "files_prefix_1.hdf5"
        ...
        - "files_prefix_files_number.hdf5"

    Parameters
    ----------
    files_prefix : str
        Name of the files to merge.
    files_number : int
        Number of files to merge.
    start_idx : int, optional
        Start index of the hdf5 files.        
    file_name : str, optional
        Name of the file with the merged data. If None, files_prefix without
        number is used.

    """
    if file_name == None:
        file_name = files_prefix
    f = hp.File(file_name + ".hdf5", "a")
    
    ## Create file architecture
    f0 = hp.File(files_prefix + "_" + str(start_idx) + ".hdf5", "r")
    for group in list(f0):
        f.require_group(group)
        for dataset_name in list(f0[group]):
            if dataset_name == "freq":
                f0[group].copy(dataset_name, f[group])
                continue
            shape = f0[group][dataset_name].shape
            dtype = f0[group][dataset_name].dtype
            shape_needed = list(shape)
            shape_needed[-1] = shape_needed[-1]*files_number
            shape_needed = tuple(shape_needed)
            f[group].create_dataset(dataset_name, shape_needed, dtype)
            
    f0.close()
    
    ## Copy data
    for i, file_num in enumerate(range(start_idx, start_idx + files_number)):
        fi = hp.File(files_prefix + "_" + str(file_num) + ".hdf5", "r")
        for group in list(fi):
            for dataset_name in list(fi[group]):
                shape = fi[group][dataset_name].shape
                n_slice = int(len(shape) - 1)
                length = shape[-1]
                slice_list = []
                for n in range(n_slice):
                    slice_list.append(slice(None))
                slice_list.append(slice(length*i,length*(i+1)))
                if (dataset_name == "freq"):
                    continue
                if (dataset_name == "time") and (file_num != start_idx):
                    f[group][dataset_name][tuple(slice_list)] = np.max(f[group][dataset_name][:]) + fi[group][dataset_name]
                else:
                    f[group][dataset_name][tuple(slice_list)] = fi[group][dataset_name]
        fi.close()
    f.close()
    
def copy_files(source, copy, version=None):
    """
    Copy a source hdf5 file into another hdf5 file using a different HDF5 
    version.
    
    The function assumes that the source file has only a single group layer.

    Parameters
    ----------
    source : str
        Name of the source file.
    copy : str
        Name of the copy file.
    version : str, optional
        Version number of the copy file.

    """
    if version == None:
        version = 'v108'
    f = hp.File(source + ".hdf5", "r")
    h = hp.File(copy + ".hdf5", "a", libver=('earliest', version))
    
    ## Copy file
    for group in list(f):
        h.require_group(group)
        for dataset_name in list(f[group]):
            h[group][dataset_name] = f[group][dataset_name][()]
            
    f.close()
    h.close()

            