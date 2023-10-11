import numpy as np
import dataclasses

@dataclasses.dataclass
class TranformerConfig:
    emb_dims: int = 512
    n_blocks: int = 1
    ff_dims: int = 1024
    n_heads: int = 4
    dropout: bool = False

@dataclasses.dataclass
class DCPConfig:
    """args:
    mutual: whether transformation from a to b and vice versa are identical
    pointer: pointer to generate correspondence
    """
    transform: TranformerConfig = TranformerConfig()
    mutual: bool = False
    pointer: str = "transformer"
    head: str = "svd" # can also be "adjugate"
    backbone: str = "dgcnn" # can also be pointnet
    