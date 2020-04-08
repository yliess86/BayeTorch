from bayetorch.layers.base import BayesianModule
from bayetorch.layers.base import EPSILON
from bayetorch.layers.base import INT_2_TWO
from bayetorch.layers.base import int_2_two
from torch import Tensor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianConv2D(BayesianModule):
    def __init__(
        self,  
        in_channels: int,
        out_channels: int,
        kernel_size: INT_2_TWO, 
        alpha_shape: INT_2_TWO = 1,
        stride: INT_2_TWO = 1, 
        padding: INT_2_TWO = 0, 
        dilation: INT_2_TWO = 1,
        groups: int = 1,
        bias: bool = True 
    ) -> None:
        super(BayesianConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size: Tuple[int, int] = int_2_two(kernel_size)
        self.alpha_shape: Tuple[int, int] = int_2_two(alpha_shape)
        self.stride: Tuple[int, int] = int_2_two(stride)
        self.padding: Tuple[int, int] = int_2_two(padding)
        self.dilation: Tuple[int, int] = int_2_two(dilation)
        self.groups = groups
        self.bias = bias

        W_size = (self.out_channels, self.in_channels, *self.kernel_size)
        self.W = nn.Parameter(Tensor(*W_size))
        self.log_alpha = nn.Parameter(Tensor(*self.alpha_shape))

        if self.bias:
            b_size = (self.out_channels, )
            self.b = nn.Parameter(Tensor(*b_size))

        self.reset()

    def reset(self) -> None:
        k1, k2 = self.kernel_size
        std = 1.0 / np.sqrt(self.in_channels * k1 * k2)
        self.W.data.uniform_(-std, std)
        self.log_alpha.data.fill_(-5.0)
        
        if self.bias:
            self.b.data.uniform_(-std, std)

    def forward(self, X: Tensor) -> Tensor:
        bias = self.b if self.bias else None
        params = (self.stride, self.padding, self.dilation, self.groups)

        mean = F.conv2d(X, self.W, bias, *params)
        sigma = torch.exp(self.log_alpha) * self.W ** 2
        std = torch.sqrt(EPSILON + F.conv2d(X ** 2, sigma, None, *params))

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
            f"BayesianConv2D("
                f"in channels: {self.in_channels}, "
                f"out channels: {self.out_channels}, "
                f"kernel size: {self.kernel_size}, "
                f"alpha shape: {self.alpha_shape}, "
                f"stride: {self.stride}, "
                f"padding: {self.padding}, "
                f"dilation: {self.padding}, "
                f"groups: {self.groups}, "
                f"has bias: {self.bias}"
            f")"
        )