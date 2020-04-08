import sys
import os

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, "../.."))

from bayetorch.layers import BayesianConv1D

import pytest
import torch


class TestConv1d:
    conv1d = BayesianConv1D(1, 2, 1, bias=True)
    
    def test_param_size(self) -> None:
        assert tuple(self.conv1d.W.size()) == (2, 1, *(1, ))
        assert tuple(self.conv1d.log_alpha.size()) == (2, 1, *(1, ))
        assert tuple(self.conv1d.b.size()) == (2, )

    def test_output_size(self) -> None:
        X = torch.zeros((32, 1, 1))
        y = self.conv1d(X)

        assert tuple(y.size()) == (32, 2, 1)

    def test_kld_size(self) -> None:
        kld = self.conv1d.kl_divergence

        assert tuple(kld.size()) == ()