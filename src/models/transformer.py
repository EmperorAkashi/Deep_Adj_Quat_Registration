import numpy as np
import torch
import torch.nn as nn

"""this file is the implementation of transformer, 
the implementation of phi in section 4.2; 
"""
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

class LayerNorm(nn.Module):
    """class to implement layer norm, which aims to shift the values 
    in each sample along the last dimension (features) 
    to have zero mean and unit variance
    """
    def __init__(self, num_features:int, eps:torch.float32 = 1e-6) -> None:
        """num_features: size of each sample
        eps: float for the Laplace smooth
        use nn.Parameter to init learnable paras
        """
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(num_features))
        self.b_2 = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class AddLayerNorm(nn.Module):
    def __init__(self, size:int, dropout=None) -> None:
        super().__init__()
        self.norm = LayerNorm(size)
    def forward(self, x:torch.Tensor, sublayer:nn.Module) -> torch.Tensor:
        return x + sublayer(self.norm(x))


class EncoderDecoder(nn.Module):
    """the core class to wrap up encode-decode architecture
    as a backbone of Transformer
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        encoded = self.encoder(src, src_mask)
        return self.decoder(encoded, src_mask, tgt, tgt_mask)