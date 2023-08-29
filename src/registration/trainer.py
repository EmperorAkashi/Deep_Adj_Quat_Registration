import logging
import hydra
import omegaconf

import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.tensorboard as tb

import pytorch_lightning as pl

import registration.config as cf

class DCPTrainer(pl.LightningModule):
    hparams: cf.DCPTrainingConfig

    def __init__(self, config: cf.DCPTrainingConfig) -> None:
        super().__init__()

        if not omegaconf.OmegaConf.is_config(config):
            config = omegaconf.OmegaConf.structured(config)
        self.save_hyperparameter(config)
        
        