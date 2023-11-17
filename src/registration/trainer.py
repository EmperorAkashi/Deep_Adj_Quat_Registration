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
        self.save_hyperparameters(config)

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

        self.training_log(batch, loss, mse_r, mse_t)
        return loss   

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

        self.validation_log(batch, loss, mse_r, mse_t)
        return loss   

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.net.parameters(), lr=self.hparams.optim.learning_rate)
        return optim    

class DCPDataModule(pl.LightningDataModule):
    def __init__(self, config: data_cf.TrainingDataConfig, batch_size:int) -> None:
        super().__init__()
        self.cf = config
        self.batch_size = batch_size

        if self.cf.option == "modelnet":
            self.ds = ds.ModelNetDataset(hydra.utils.to_absolute_path(self.cf.file_path), 
                                    self.cf.config.category, self.cf.config.num_points, 
                                    self.cf.config.sigma,self.cf.config.num_rot,
                                    self.cf.config.range_max, self.cf.config.range_min,
                                    self.cf.config.rot_option, 
                                    self.cf.config.trans_max, self.cf.config.rot_max)
        
        self.ds_train = None
        self.ds_val = None

    def setup(self, stage: str = None) -> None:
        if self.cf.limit is not None:
            limit = min(self.cf.limit, len(self.ds))
            self.ds, _ = torch.utils.data.random_split(self.ds, [limit, len(self.ds) - limit])

        num_train_samples = int(len(self.ds) * self.cf.train_prop)

        self.ds_train, self.ds_val = torch.utils.data.random_split(
            self.ds, [num_train_samples, len(self.ds) - num_train_samples],
            torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.ds_train, self.batch_size, shuffle=True, num_workers=self.cf.num_data_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.ds_val, self.batch_size, shuffle=False, num_workers=self.cf.num_data_workers)


@hydra.main(config_path=None, config_name='train', version_base='1.1' ) 
def main(config: cf.DCPTrainingConfig):
    logger = logging.getLogger(__name__)
    trainer = pl.Trainer(
        accelerator=config.device, 
        devices=config.num_gpus,
        log_every_n_steps=config.log_every,
        max_epochs=config.num_epochs)
    
    data_config = config.data
    dm = DCPDataModule(data_config, config.batch_size)
    model = DCPTrainer(config)

    trainer.fit(model,dm)

    if trainer.is_global_zero:
        logger.info(f'Finished training. Final MSE loss: {trainer.logged_metrics["train/mse_loss"]}')
        logger.info(f'Finished training. Final MSE of Rotation: {trainer.logged_metrics["train/mse_rot"]}')
        logger.info(f'Finished training. Final MSE of Translation: {trainer.logged_metrics["train/mse_trans"]}')


if __name__ == '__main__':
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore()
    cs.store('train', node=cf.DCPTrainingConfig)
    main()

        