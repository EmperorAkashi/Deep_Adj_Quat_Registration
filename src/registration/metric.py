import numpy as np
import torch
from enum import Enum
from abc import ABC, abstractmethod

class Loss(Enum):
    mse = 1

class LossFn(ABC):
    @abstractmethod
    def compute_loss(self, rot_ab:torch.Tensor, trans_ab:torch.Tensor, 
                    rot_ba:torch.Tensor, trans_ba:torch.Tensor,
                    rot_pred:torch.Tensor, trans_pred:torch.Tensor) -> torch.Tensor:
        pass

class MSELoss(LossFn):
    def compute_loss(self, rot_ab:torch.Tensor, trans_ab:torch.Tensor, 
                    rot_ba:torch.Tensor, trans_ba:torch.Tensor,
                    rot_pred:torch.Tensor, trans_pred:torch.Tensor) -> torch.Tensor:
        batch, _, _ = rot_ab.shape
        identity = torch.eye(3).unsqueeze(0).repeat(batch,1,1).to(rot_ab.device)

        mse = torch.nn.MSELoss()
        loss = mse(torch.matmul(rot_ab,rot_pred),identity) + mse(trans_ab,trans_pred)
        return loss

class LossFactory:
    def create(self, loss_name):
        switcher = {
            'mse': MSELoss
        }
        return switcher.get(loss_name)