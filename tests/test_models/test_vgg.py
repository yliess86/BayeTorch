import sys
import os

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, "../.."))

from bayetorch.models import BayesianVGG11
from bayetorch.models import BayesianVGG13
from bayetorch.models import BayesianVGG16
from bayetorch.models import BayesianVGG19

import pytest
import torch


class TestVGG:
    def test_vgg11(self) -> None:
        model = BayesianVGG11(2)
        
        X = torch.zeros((32, 3, 224, 224))
        y, kld = model(X)

        assert tuple(y.size()) == (32, 2)
        assert tuple(kld.size()) == ()

    def test_vgg13(self) -> None:
        model = BayesianVGG13(2)
        
        X = torch.zeros((32, 3, 224, 224))
        y, kld = model(X)

        assert tuple(y.size()) == (32, 2)
        assert tuple(kld.size()) == ()
        
    def test_vgg16(self) -> None:
        model = BayesianVGG16(2)
        
        X = torch.zeros((32, 3, 224, 224))
        y, kld = model(X)

        assert tuple(y.size()) == (32, 2)
        assert tuple(kld.size()) == ()

    def test_vgg19(self) -> None:
        model = BayesianVGG19(2)
        
        X = torch.zeros((32, 3, 224, 224))
        y, kld = model(X)

        assert tuple(y.size()) == (32, 2)
        assert tuple(kld.size()) == ()