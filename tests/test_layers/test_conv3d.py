import sys
import os

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, "../.."))

from bayetorch.layers import BayesianConv3D

import pytest
import torch


class TestConv3d:
    conv3d = BayesianConv3D(1, 2, 1, bias=True)
    
    def test_param_size(self) -> None:
        assert tuple(self.conv3d.W.size()) == (2, 1, *(1, 1, 1))
        assert tuple(self.conv3d.log_alpha.size()) == (2, 1, *(1, 1, 1))
        assert tuple(self.conv3d.b.size()) == (2, )

    def test_output_size(self) -> None:
        X = torch.zeros((32, 1, 1, 1, 1))
        y = self.conv3d(X) 

        assert tuple(y.size()) == (32, 2, 1, 1, 1)

    def test_kld_size(self) -> None:
        kld = self.conv3d.kl_divergence

        assert tuple(kld.size()) == ()