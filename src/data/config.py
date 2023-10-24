import dataclasses
from typing import Optional, Tuple, List, Type, Union
import omegaconf
from enum import Enum
    
@dataclasses.dataclass
class BaseConfig:
    pass
@dataclasses.dataclass
class ModelNetConfig(BaseConfig):
    category: List[str] = dataclasses.field(default_factory=lambda:["airplane"])
    sigma: float = 0.01
    num_points: int = 1000 #downsampled size for the modelnet mesh
    num_rot: int = 1000
    range_max: int = 35000
    range_min: int = 30000
    trans_max: float = 0.5
    rot_max: int = 45
    rot_option: str = "JPL" #can be Hamitonian with our custom code

@dataclasses.dataclass
class KITTIConfig:
    pass

CONFIG_MAP = {
    "modelnet": ModelNetConfig
    }

@dataclasses.dataclass
class TrainingDataConfig(BaseConfig):
    """configuration of data loading

    attr:
    file_path(str): path of simulated point cloud and quaternion
    train_prop(float): percentage for training
    """
    file_path: str = omegaconf.MISSING
    option: str = omegaconf.MISSING 
    config: BaseConfig = ModelNetConfig()
    train_prop: float = 0.9
    limit: Optional[int] = None
    num_data_workers: int = 16
    svd_mod: bool = False #use qrmsd or qinit



