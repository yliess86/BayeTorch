from bayetorch.layers.base import BayesianModule
from bayetorch.layers.base import EPSILON
from torch import Tensor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianLinear(BayesianModule):
    def __init__(
        self,  
        in_features: int, 
        out_features: int,
        bias: bool = True
    ) -> None:
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        W_size = (self.out_features, self.in_features)
        self.W = nn.Parameter(Tensor(*W_size))
        self.log_alpha = nn.Parameter(Tensor(*self.W.size()))
        
        if self.bias:
            b_size = (self.out_features, )
            self.b = nn.Parameter(Tensor(*b_size))

        self.reset()

    def reset(self) -> None:
        std = 1.0 / np.sqrt(self.W.size(1))
        self.W.data.uniform_(-std, std)
        self.log_alpha.data.fill_(-5.0)
        
        if self.bias:
            self.b.data.zero_()

    def forward(self, X: Tensor) -> Tensor:
        mean = F.linear(X, self.W) + (self.b if self.bias else 0.0)
        sigma = torch.exp(self.log_alpha) * self.W ** 2
        std = torch.sqrt(EPSILON + F.linear(X ** 2, sigma))
        
        X = self.reparametrize(mean, std)

        return X

    @property
    def kl_divergence(self) -> Tensor:
        alpha = torch.exp(-self.log_alpha)
        kld_log_alpha = 0.5 * torch.sum(torch.log1p(alpha))
        kld = self.W.nelement() / self.log_alpha.nelement() * kld_log_alpha

        return kld

    def __str__(self) -> str:
        return (
            f"BayesianLinear("
                f"in features: {self.in_features}, "
                f"out features: {self.out_features}, "
                f"has bias: {self.bias}"
            f")"
        )