import sys
import os

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, ".."))

from bayetorch.models import BayesianLeNet5
from bayetorch.models import BayesianVGG11
from bayetorch.models import BayesianVGG13
from bayetorch.models import BayesianVGG16
from bayetorch.models import BayesianVGG19
from bayetorch.models import LeNet5
from bayetorch.models import VGG11
from bayetorch.models import VGG13
from bayetorch.models import VGG16
from bayetorch.models import VGG19

import pytest
import torch.nn as nn


class TestInitBayesianWithFrequetist:
    def assertions(
        self,
        bayesian: nn.Module,
        frequentist: nn.Module,
        freeze: bool
    ) -> None:
        bayesian_named_params = bayesian.named_parameters()
        frequentist_named_params = frequentist.named_parameters()

        filtr = lambda data: "log_alpha" not in data[0]

        bayesian_named_params = filter(filtr, bayesian_named_params)
        frequentist_named_params = filter(filtr, frequentist_named_params)
        bayesian_named_params = list(bayesian_named_params)
        frequentist_named_params = list(frequentist_named_params)

        assert len(bayesian_named_params) == len(frequentist_named_params)

        named_params = zip(bayesian_named_params, frequentist_named_params)
        for (_, bayesian_param), (_, frequentist_param) in named_params:
            assert (bayesian_param.data == frequentist_param.data).all()
            assert bayesian_param.requires_grad == (not freeze)

    def test_lenet(self) -> None:
        freq = LeNet5(n_classes=10)
        baye = BayesianLeNet5(n_classes=10)

        baye = baye.init_with_frequentist(freq, True)
        self.assertions(baye, freq, True)

        baye = baye.init_with_frequentist(freq, False)
        self.assertions(baye, freq, False)

    def test_vgg11(self) -> None:
        freq = VGG11(n_classes=10)
        baye = BayesianVGG11(n_classes=10)

        baye = baye.init_with_frequentist(freq, True)
        self.assertions(baye, freq, True)

        baye = baye.init_with_frequentist(freq, False)
        self.assertions(baye, freq, False)

    def test_vgg13(self) -> None:
        freq = VGG13(n_classes=10)
        baye = BayesianVGG13(n_classes=10)

        baye = baye.init_with_frequentist(freq, True)
        self.assertions(baye, freq, True)

        baye = baye.init_with_frequentist(freq, False)
        self.assertions(baye, freq, False)

    def test_vgg16(self) -> None:
        freq = VGG16(n_classes=10)
        baye = BayesianVGG16(n_classes=10)

        baye = baye.init_with_frequentist(freq, True)
        self.assertions(baye, freq, True)

        baye = baye.init_with_frequentist(freq, False)
        self.assertions(baye, freq, False)

    def test_vgg19(self) -> None:
        freq = VGG19(n_classes=10)
        baye = BayesianVGG19(n_classes=10)

        baye = baye.init_with_frequentist(freq, True)
        self.assertions(baye, freq, True)

        baye = baye.init_with_frequentist(freq, False)
        self.assertions(baye, freq, False)