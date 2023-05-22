import numpy as np
import torch
import torch.nn as nn

def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
              mask=None, dropout=None) -> torch.Tensor:
    
    """
    return: normalized score map, softmax(q*k) and attention map
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2,-1).contiguous())/torch.sqrt(d_k)

    if mask:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = nn.functional.softmax(scores, dim=-1)

    return torch.matmul(p_attn, value), p_attn

def nearest_neighbor(src: torch.Tensor, dst: torch.Tensor):
    """
    calculate the smallest distanced point from dst to src
    args: src & dst are two tensors with dim
    dim of feature*num points
    broadcasting the complete square, i.e. the first step calculated 2ab
    """
    inner = -2*torch.matmul(src.transpose(1,0).contiguous(), dst)

    dist = -torch.sum(src**2, dim=0, keepdim=True).transpose(1,0).contiguous()
    - inner - torch.sum(dst**2, dim=0, keepdim=True)

    dist, idx = dist.topk(k=1, dim=-1)
    return dist, idx

def knn(cloud:torch.Tensor, k:int) -> torch.Tensor:
    """
    calculate the top k nearest point from same cloud
    args: cloud
    the point cloud to be calculated
    """
    inner = -2*torch.matmul(cloud.transpose(2,1).contiguous(), cloud)
    cloud_sqr = torch.sum(cloud**2, dim=1, keepdim=True)

    pairwise = -cloud_sqr - inner - cloud_sqr.transpose(2,1).contiguous()

    idx = pairwise.topk(k=k, dim=1)[1]
    return idx

