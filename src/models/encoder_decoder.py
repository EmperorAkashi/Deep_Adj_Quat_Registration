import numpy as np
import torch
import torch.nn as nn

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