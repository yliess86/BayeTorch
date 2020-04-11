from bayetorch.metrics import ELBO
from typing import Tuple
from torch import Tensor
from torch.optim import Optimizer
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


def step_frequentist(
    model: nn.Module,
    X: Tensor,
    Y: Tensor,
    n: int,
    optim: Optimizer,
    criterion: nn.Module, 
    test: bool
) -> Tuple[float, float]:
    if not test:
        optim.zero_grad()
    X, Y = X.float().cuda(), Y.long().cuda()

    y = model(X)
    loss = criterion(y, Y)

    if not test:
        loss.backward()
        optim.step()

    acc = torch.argmax(y, axis=-1).eq(Y).sum().float() / n

    return loss.item(), acc.item()


def step_bayesian(
    model: nn.Module,
    X: Tensor,
    Y: Tensor,
    n: int,
    optim: Optimizer,
    criterion: nn.Module,
    samples: int,
    test: bool
) -> Tuple[float, float]:
    if not test:
        optim.zero_grad()
    X, Y = X.float().cuda(), Y.long().cuda()

    ys_size = (X.size(0), model.n_classes, samples)
    ys = torch.zeros(ys_size).cuda()
    klds = 0.0

    for sample in range(samples):
        y, kld = model(X)
        ys[:, :, sample] = F.log_softmax(y, dim=1)
        klds += kld

    log_y = ELBO.log_mean_exp(ys, dim=2)
    loss = criterion(log_y, Y, klds)

    if not test:
        loss.backward()
        optim.step()

    acc = torch.argmax(log_y, axis=-1).eq(Y).sum().float() / n

    return loss.item(), acc.item()