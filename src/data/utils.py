import numpy as np
import glob
import h5py
import os

import torch.nn as nn

def center_norm(loaded_cloud:np.ndarray) -> np.ndarray:
    point_set = loaded_cloud - np.expand_dims(np.mean(loaded_cloud, axis = 0), 0) # centering
    dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
    point_set = point_set / dist #normalization

    return point_set

def load_shapenet(base_path:str, partition:str) -> None:
    data_dir = os.path.join(base_path, 'data')

    all_data = []
    all_label = []

    for h5 in glob.glob(os.path.join(data_dir, 
                        'modelnet40_ply_hdf5_2048', 
                        'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)

    return all_data, all_label