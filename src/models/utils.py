import numpy as np
import torch
import torch.nn as nn
import copy

def clones(module:nn.Module, N:int) -> nn.ModuleList:
    """utility function to have deep copied instances
    for a specific nn.Module
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])