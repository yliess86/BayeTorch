import sys
import os

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, ".."))

from bayetorch.metrics import ELBO
from bayetorch.models import BayesianLeNet5
from torch.optim import Adam

import pytest
import torch
import torch.nn.functional as F


class TestELBO:
    def test_elbo(self):
        criterion = ELBO(n_samples=32)

        predictions = torch.zeros((32, 2)).float()
        klds = torch.zeros((32, 1)).float()
        targets = torch.zeros((32)).long()

        loss = criterion(predictions, targets, klds)
        assert loss.sum() == 0.0

    def test_lenet_mnist_elbo(self):
        images = torch.rand((64, 1, 28, 28)).float()
        labels = torch.rand((64)).long()
        loader = [
            (images[i * 32 : i * 32 + 32], labels[i * 32 : i * 32 + 32])
            for i in range(64 // 32)
        ]

        model = BayesianLeNet5(10)
        criterion = ELBO(n_samples=64)
        optim = Adam(model.parameters(), lr=1e-3)

        for X, Y in loader:
            optim.zero_grad()

            y, kld = model(X)
            log_y = F.log_softmax(y, dim=-1)
            loss = criterion(log_y, Y, kld)

            loss.backward()
            optim.step()