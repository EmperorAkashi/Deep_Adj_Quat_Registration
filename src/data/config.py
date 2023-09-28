import dataclasses
from typing import Optional, Tuple, List, Type, Union
import omegaconf
from enum import Enum
    

@dataclasses.dataclass
class ModelNetConfig:
    category: List[str] = dataclasses.field(default_factory=lambda:["airplane"])
    sigma: float = 0.01
    num_points: int = 1000 #downsampled size for the modelnet mesh
    num_rot: int = 1000
    range_max: int = 35000
    range_min: int = 30000
    trans_min: float = -0.5
    trans_max: float = 0.5

@dataclasses.dataclass
class KITTIConfig:
    pass

CONFIG_MAP = {
    "modelnet": ModelNetConfig
    }

@dataclasses.dataclass
class TrainingDataConfig:
    """configuration of data loading

    attr:
    file_path(str): path of simulated point cloud and quaternion
    train_prop(float): percentage for training
    """
    config: Union[ModelNetConfig, KITTIConfig] = dataclasses.field(lambda:ModelNetConfig())  # Avoid automatic initialization
    file_path: str = omegaconf.MISSING 
    train_prop: float = 0.9
    limit: Optional[int] = None
    num_data_workers: int = 16
    svd_mod: bool = False #use qrmsd or qinit
    option: str = omegaconf.MISSING

    def __init__(self, option: str):
        self.dataset = option
        self.config = CONFIG_MAP[self.dataset]()