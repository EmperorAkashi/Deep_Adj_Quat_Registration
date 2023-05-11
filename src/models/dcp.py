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

