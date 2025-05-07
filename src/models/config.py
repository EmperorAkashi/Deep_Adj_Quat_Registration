import dataclasses

@dataclasses.dataclass
class TransformerConfig:
    emb_dims: int = 512
    n_blocks: int = 1 # number of encoder layers
    ff_dims: int = 1024 # number of hidden units, i.e. number of points in the cloud
    n_heads: int = 4 # number of heads of multi-attention
    dropout: bool = False

@dataclasses.dataclass
class DCPConfig:
    """args:
    mutual: whether transformation from a to b and vice versa are identical
    pointer: pointer to generate correspondence
    """
    transform: TransformerConfig = TransformerConfig()
    mutual: bool = False
    pointer: str = "transformer"
    head: str = "svd" # can also be "adjugate"
    backbone: str = "dgcnn" # can also be pointnet
    