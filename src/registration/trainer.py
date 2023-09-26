import logging
import hydra
import omegaconf

import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.tensorboard as tb

import pytorch_lightning as pl

import registration.config as cf
import data.config as data_cf
import data.dataset as ds

from models.dcp import DCP
import registration.metric as M
import utils.quat_util as Q

class DCPTrainer(pl.LightningModule):
    hparams: cf.DCPTrainingConfig

    def __init__(self, config: cf.DCPTrainingConfig) -> None:
        super().__init__()

        if not omegaconf.OmegaConf.is_config(config):
            config = omegaconf.OmegaConf.structured(config)
        self.save_hyperparameter(config)

        self.net = DCP(config.model)
        self.config = config

    def forward(self, x):
        return self.net(x)

    def training_log(self, batch, loss:float, mse_r:float, mse_t:float):
        cloud, quat, trans = batch
        self.log('train/mse_loss', loss)
        self.log('train/mse_rot', mse_r)
        self.log('train/mse_trans', mse_t)

    def training_step(self, batch, batch_idx: int):
        cloud, quat, trans = batch
        rot_ab_pred, trans_ab_pred, _, _ = self(cloud)

        rot = Q.batch_quat_to_rot(quat)
        
        loss_create = M.LossFactory()
        loss_computer = loss_create.create(self.config.loss)

        loss, mse_r, mse_t = loss_computer.compute_loss(rot, trans, rot_ab_pred, trans_ab_pred)

        return loss, mse_r, mse_t    

    def validation_log(self, batch, loss:float, mse_r:float, mse_t:float):
        cloud, quat, trans = batch
        self.log('val/mse_loss', loss)
        self.log('val/mse_rot', mse_r)
        self.log('val/mse_trans', mse_t)

    def validation_step(self, batch, batch_idx: int):
        cloud, quat, trans = batch
        rot_ab_pred, trans_ab_pred, _, _ = self(cloud)

        rot = Q.batch_quat_to_rot(quat)
        
        loss_create = M.LossFactory()
        loss_computer = loss_create.create(self.config.loss)

        loss, mse_r, mse_t = loss_computer.compute_loss(rot, trans, rot_ab_pred, trans_ab_pred)

        return loss, mse_r, mse_t    

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.point_net.parameters(), lr=self.hparams.optim.learning_rate)
        return optim    

class DCPDataModule(pl.LightningDataModule):
    def __init__(self, config: data_cf.TrainingDataConfig, batch_size:int) -> None:
        super().__init__()
        self.cf = config
        self.batch_size = batch_size

        if self.config.option == "modelnet":
            self.ds = ds.ModelNetDataset(hydra.utils.to_absolute_path(self.cf.file_path), 
                                    self.cf.config.category, self.cf.config.num_points, 
                                    self.cf.config.sigma,self.cf.config.num_rot,
                                    self.cf.config.range_max, self.cf.config.range_min)




    

        