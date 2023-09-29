import torch
import torch.nn as nn
import models.config as cf
import models.knn as K

"""
class to implement Dynamic Graphic CNN;
The input is raw cloud, which will be formulated as Knn graph first
"""
class DGCNN(nn.Module):
    def __init__(self, emb_dims:int = 512) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(emb_dims)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        batch_size, num_dims, num_points = x.size()
        # get k nearest graph for the batch of clouds
        # will in a shape (batch_size, num_dims, num_points, k)
        x = K.get_graph_features(x)

        # get most significant feature along k (dim=-1)
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = nn.functional.relu(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=True)[0]

        # cat along the channel
        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = nn.functional.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
        return x


