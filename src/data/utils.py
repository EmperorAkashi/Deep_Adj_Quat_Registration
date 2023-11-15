import numpy as np
import torch
import torch.nn as nn

def center_norm(loaded_cloud:np.ndarray) -> np.ndarray:
    point_set = loaded_cloud - np.expand_dims(np.mean(loaded_cloud, axis = 0), 0) # centering
    dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
    point_set = point_set / dist #normalization

    return point_set