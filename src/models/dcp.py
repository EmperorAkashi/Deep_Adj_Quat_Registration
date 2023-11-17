import torch
import torch.nn as nn
import models.config as cf
from models.dgcnn import DGCNN
import models.head as H
from models.transformer import Transformer

class DCP(nn.Module):
    # TO DO: both head and backbone should be wrap up as a factory pattern
    def __init__(self, cf:cf.DCPConfig) -> None:
        super().__init__()
        self.emb_dims = cf.transform.emb_dims
        self.mutual = cf.mutual

        if cf.backbone == "dgcnn":
            self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        else:
            raise Exception("Not a valid backbone")

        if cf.pointer == "transformer":
            self.pointer = Transformer(cf.transform)


        if cf.head == "svd":
            self.head = H.SVDHead(self.emb_dims)

    def forward(self, input_data:torch.Tensor) -> torch.Tensor:
        """the transformer mechanism generate a relative probability 
        map for the correspondence x and y;
        by multiplying this map, 
        it shift the raw cloud y to a matched averaged point y_hat
        """
        src = input_data[:,0,:,:]
        tgt = input_data[:,1,:,:]

        src_embedding = self.emb_nn(src)
        tgt_embedding = self.emb_nn(tgt)


        src_embedding_p, tgt_embedding_p = self.pointer(src_embedding, tgt_embedding)


        src_embedding = src_embedding + src_embedding_p
        tgt_embedding = tgt_embedding + tgt_embedding_p

        rotation_ab, translation_ab = self.head(src_embedding, tgt_embedding, src, tgt)
        if self.mutual:
            rotation_ba, translation_ba = self.head(tgt_embedding, src_embedding, tgt, src)

        else:
            rotation_ba = rotation_ab.transpose(2, 1).contiguous()
            translation_ba = -torch.matmul(rotation_ba, translation_ab.unsqueeze(2)).squeeze(2)
        return rotation_ab, translation_ab, rotation_ba, translation_ba
