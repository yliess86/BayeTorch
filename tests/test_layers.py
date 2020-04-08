import sys
import os

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, ".."))

from bayetorch.layers import BayesianConv2D
from bayetorch.layers import BayesianLinear

import pytest
import torch


class TestLinear:
    linear = BayesianLinear(1, 2, 1, bias=True)
    
    def test_param_size(self) -> None:
        assert tuple(self.linear.W.size()) == (2, 1)
        assert tuple(self.linear.b.size()) == (2, )

    def test_output_size(self) -> None:
        X = torch.zeros((32, 1))
        y = self.linear(X) 

        assert tuple(y.size()) == (32, 2)

    def test_kld_size(self) -> None:
        kld = self.linear.kl_divergence

        assert tuple(kld.size()) == ()


class TestConv2d:
    conv2d = BayesianConv2D(1, 2, 1, 1, bias=True)
    
    def test_param_size(self) -> None:
        assert tuple(self.conv2d.W.size()) == (2, 1, *(1, 1))
        assert tuple(self.conv2d.b.size()) == (2, )

    def test_output_size(self) -> None:
        X = torch.zeros((32, 1, 1, 1))
        y = self.conv2d(X) 

        assert tuple(y.size()) == (32, 2, 1, 1)

    def test_kld_size(self) -> None:
        kld = self.conv2d.kl_divergence

        assert tuple(kld.size()) == ()