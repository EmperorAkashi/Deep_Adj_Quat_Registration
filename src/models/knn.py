import numpy as np
import torch
import torch.nn as nn

def nearest_neighbor(src: torch.Tensor, dst: torch.Tensor):
    """
    calculate the smallest distanced point from dst to src
    args: src & dst are two tensors with dim
    dim of feature*num points
    broadcasting the complete square (a+b)^2, i.e. the first step calculated 2ab
    """
    inner = -2*torch.matmul(src.transpose(1,0).contiguous(), dst)

    dist = -torch.sum(src**2, dim=0, keepdim=True).transpose(1,0).contiguous()
    - inner - torch.sum(dst**2, dim=0, keepdim=True)

    dist, idx = dist.topk(k=1, dim=-1)
    return dist, idx

def knn(cloud:torch.Tensor, k:int) -> torch.Tensor:
    """
    calculate the indices of top k nearest point 
    for each point from the same cloud
    args: cloud
    the point cloud to be calculated
    """
    inner = -2*torch.matmul(cloud.transpose(2,1).contiguous(), cloud)
    cloud_sqr = torch.sum(cloud**2, dim=1, keepdim=True)

    pairwise = -cloud_sqr - inner - cloud_sqr.transpose(2,1).contiguous()

    #should be (batch_size, num_points, k)
    idx = pairwise.topk(k=k, dim=1)[1]
    return idx

def get_graph_features(cloud:torch.Tensor, k:int = 20) -> torch.Tensor:
    """get k-nearest features for each point in one cloud
    args: cloud in shape (b,n_dim,n_point)
    in original DCP's implementation, it first generate idx_base in shape (batch_size, 1, 1)
    with values [0, num_points, 2*num_points, ..., (batch_size-1)*num_points]
    then shift idx by idx_base; this is because it will flatten the x as (batch_size * num_points, -1)
    then use the shifted idx to indexing corresponding entries.
    """
    idx = knn(cloud, k=k) # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()
    _, num_dims, _ = cloud.size()

    # Generate rows with shape (batch_size, 1, 1)
    rows = torch.arange(batch_size, device=cloud.device).view(-1, 1, 1) 
    cols = torch.arange(num_points, device=cloud.device).view(1, -1, 1)

    feature = cloud[rows, idx, :]  # (batch_size, num_points, k, num_dims)
    cloud = cloud.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature, cloud), dim=3).permute(0, 3, 1, 2)

    return feature