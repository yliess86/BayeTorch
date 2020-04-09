import sys
import os

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, "../.."))

from bayetorch.models import BayesianLeNet5

import pytest
import torch


class TestLeNet5:
    def test_lenet(self) -> None:
        model = BayesianLeNet5(2)
        
        X = torch.zeros((32, 1, 28, 28))
        y, kld = model(X)

        assert tuple(y.size()) == (32, 2)
        assert tuple(kld.size()) == ()
        assert not torch.isnan(y).any()
        assert not torch.isnan(kld).any()