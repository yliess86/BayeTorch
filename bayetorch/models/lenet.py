from bayetorch.layers.conv2d import BayesianConv2D
from bayetorch.layers.linear import BayesianLinear
from bayetorch.models.base import BayesianModel
from torch import Tensor
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianLeNet5(BayesianModel):
    def __init__(self, n_classes: int = 10) -> None:
        super(BayesianLeNet5, self).__init__()
        self.conv1 = BayesianConv2D(1,  6, 5)
        self.conv2 = BayesianConv2D(6, 16, 5)
        
        self.fc1 = BayesianLinear(256,       120)
        self.fc2 = BayesianLinear(120,        84)
        self.fc3 = BayesianLinear( 84, n_classes)

    def forward(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        X = torch.max_pool2d(F.softplus(self.conv1(X)), 2)
        X = torch.max_pool2d(F.softplus(self.conv2(X)), 2)

        X = X.view(X.size(0), -1)
        
        X = F.softplus(self.fc1(X))
        X = F.softplus(self.fc2(X))
        X = self.fc3(X)
        
        kld = self.kl_divergence

        return X, kld