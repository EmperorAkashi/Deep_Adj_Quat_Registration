import numpy as np
import torch
import torch.nn as nn
from models.utils import clones

"""this file is the implementation of transformer, 
the implementation of phi in section 4.2; 
"""

def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
              mask=None, dropout=None) -> torch.Tensor:
    
    """
    return: normalized score map, softmax(q*k) and attention map
    mask is the map of the opposite sequence (i.e. x v.s. y), to
    mask the attention weights conditional on y or x
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
    """equivalent to SubLayerConnect which shift x via
    attn map or other nn modules
    """
    def __init__(self, size:int, dropout=None) -> None:
        super().__init__()
        self.norm = LayerNorm(size)
    def forward(self, x:torch.Tensor, sublayer:nn.Module) -> torch.Tensor:
        return x + sublayer(self.norm(x))

class EncoderLayer(nn.Module):
    """regular encoder layer with first apply self-attention,
    then apply feed forward layer; each have separate add norm layer 
    to shift the resulted tensor as a normalized one.
    Separate instances of layer norms are used due to all instances 
    have separate learnable parameters
    """
    def __init__(self, size:int, attn:nn.Module, feed_forward:nn.Module, dropout=None) -> None:
        super().__init__()
        self.attn = attn
        self.ff = feed_forward
        self.sublayer = clones(AddLayerNorm(size, dropout),2)
        self.size = size #will be called by the interface

    def forward(self, x:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
        x = self.sublayer[0](x, lambda x: self.attn(x, x, x, mask))
        return self.sublayer[1](x. self.ff)

class DecoderLayer(nn.Module):
    """second decode layer uses opposite map as key 
    in the attention module, i.e. x to y and vice versa
    """
    def __init__(self, size:int, attn:nn.Module, src_attn:nn.Module, feed_forward:nn.Module, dropout) -> None:
        super().__init__()
        self.attn = attn
        self.src_attn = src_attn
        self.ff = feed_forward
        self.sublayer = clones(AddLayerNorm(size, dropout), 3)
        self.size = size

    def forward(self, x:torch.Tensor, memory:torch.Tensor, src_mask:torch.Tensor, 
                tgt_mask:torch.Tensor) -> torch.Tensor:
        """memory is equivalent to the target feature map;
        in the case of src attention, the key and value are different
        from the source map, which can be regarded as a cross-correlation
        or conditional attention
        """
        m = memory
        x = self.sublayer[0](x, lambda x: self.attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, tgt_mask))
        return self.sublayer[2](x, self.ff)

class Encoder(nn.Module):
    """class interface to implement multi-layer encoder
    """
    def __init__(self, layer:nn.Module, N:int) -> None:
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    def forward(self, x:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)

class Decoder(nn.Module):
    """class interface to implement multi-layer decoder
    """
    def __init__(self, layer:nn.Module, N:int) -> None:
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x:torch.Tensor, memory:torch.Tensor, src_mask:torch.Tensor, 
                tgt_mask:torch.Tensor) -> torch.Tensor:
        "we do not have mask in the real implementation"
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class EncoderDecoder(nn.Module):
    """the core class to wrap up encode-decode architecture
    as a backbone of Transformer; src&tgt embed are optional
    embedding after DGCNN&before encoder/decoder, we set them to
    None by default
    We also skip the final generator layer of the "vanilla" Transformer
    as we will use special "head" to generate rot/trans
    """
    def __init__(self, encoder:nn.Module, decoder:nn.Module, src_embed:nn.Module=None, 
                tgt_embed:nn.Module=None) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed

    def forward(self, src:torch.Tensor, tgt:torch.Tensor, 
                src_mask:torch.Tensor, tgt_mask:torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(src, src_mask)
        decoded =  self.decoder(encoded, tgt, src_mask, tgt_mask)
        return decoded

class MultiHeadAttn(nn.Module):
    def __init__(self, feature_dim:int, num_heads:int) -> None:
        super().__init__()
        assert feature_dim % num_heads == 0
        self.d_k = feature_dim//num_heads
        self.n_dim = feature_dim
        self.n_head = num_heads
        self.linears = clones(nn.Linear(feature_dim, feature_dim), 4)
        self.attn = None
        self.dropout = None