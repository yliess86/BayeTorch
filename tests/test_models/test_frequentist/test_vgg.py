import sys
import os

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, "../.."))

from bayetorch.models import VGG11
from bayetorch.models import VGG13
from bayetorch.models import VGG16
from bayetorch.models import VGG19

import pytest
import torch


class TestVGG:
    def test_vgg11(self) -> None:
        model = VGG11(2)
        
        X = torch.zeros((32, 3, 32, 32))
        y = model(X)

        assert tuple(y.size()) == (32, 2)
        assert not torch.isnan(y).any()

    def test_vgg13(self) -> None:
        model = VGG13(2)
        
        X = torch.zeros((32, 3, 32, 32))
        y = model(X)

        assert tuple(y.size()) == (32, 2)
        assert not torch.isnan(y).any()
        
    def test_vgg16(self) -> None:
        model = VGG16(2)
        
        X = torch.zeros((32, 3, 32, 32))
        y = model(X)

        assert tuple(y.size()) == (32, 2)
        assert not torch.isnan(y).any()

    def test_vgg19(self) -> None:
        model = VGG19(2)
        
        X = torch.zeros((32, 3, 32, 32))
        y = model(X)

        assert tuple(y.size()) == (32, 2)
        assert not torch.isnan(y).any()