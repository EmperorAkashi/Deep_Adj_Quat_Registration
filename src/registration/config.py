import dataclasses
from typing import Optional, List
import omegaconf

import models.config as m_cf
import data.config as d_cf

@dataclasses.dataclass
class OptimConfig:
    """hyperparams of optimization

    attr:
    learning_rate(float): lr of training
    """
    learning_rate: float = 1e-3

@dataclasses.dataclass
class DCPTrainingConfig:
    data: d_cf.TrainingDataConfig = d_cf.TrainingDataConfig()
    optim: OptimConfig = OptimConfig()
    model: m_cf.DCPConfig = m_cf.DCPConfig()
    batch_size: int = 64
    num_epochs: int = 10
    device: str = 'gpu'
    num_gpus: int = 1
    log_every: int = 1


