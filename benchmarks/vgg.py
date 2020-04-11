import os
import sys

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, ".."))

from bayetorch.metrics import ELBO
from bayetorch.models import BayesianVGG11
from bayetorch.models import BayesianVGG16
from bayetorch.models import VGG11
from bayetorch.models import VGG16
from benchmarks.base import Benchmark
from benchmarks.base import SizeEstimator
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import RandomCrop
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import ToTensor
from tqdm import tqdm
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BenchmarkVGG11(Benchmark):
    def __init__(
        self,
        epochs: int,
        batch_size: int,
        n_workers: int,
        root: str,
        f_lr: float,
        b_lr: float,
        samples: int
    ) -> None:
        super(BenchmarkVGG11, self).__init__(
            epochs, batch_size, n_workers, root, f_lr, b_lr, samples
        )

        train_transform = Compose([ 
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010))
        ])
        valid_transform = Compose([
            ToTensor(),
            Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010))
        ])

        self.train_loader = DataLoader(
            CIFAR10(self.root, True, transform=train_transform, download=True),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers,
            pin_memory=True
        )
        self.valid_loader = DataLoader(
            CIFAR10(self.root, False, transform=valid_transform, download=True),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers,
            pin_memory=True
        )

        self.frequentist = VGG11(n_classes=10)
        self.bayesian = BayesianVGG11(n_classes=10)


class BenchmarkVGG16(Benchmark):
    def __init__(
        self,
        epochs: int,
        batch_size: int,
        n_workers: int,
        root: str,
        f_lr: float,
        b_lr: float,
        samples: int
    ) -> None:
        super(BenchmarkVGG16, self).__init__(
            epochs, batch_size, n_workers, root, f_lr, b_lr, samples
        )

        train_transform = Compose([ 
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010))
        ])
        valid_transform = Compose([
            ToTensor(),
            Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010))
        ])

        self.train_loader = DataLoader(
            CIFAR10(self.root, True, transform=train_transform, download=True),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers,
            pin_memory=True
        )
        self.valid_loader = DataLoader(
            CIFAR10(self.root, False, transform=valid_transform, download=True),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers,
            pin_memory=True
        )

        self.frequentist = VGG16(n_classes=10)
        self.bayesian = BayesianVGG16(n_classes=10)