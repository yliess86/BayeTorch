import os
import sys

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, ".."))

from bayetorch.models import BayesianLeNet5
from bayetorch.models import LeNet5
from benchmarks.base import Benchmark
from benchmarks.base import SizeEstimator
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor


class BenchmarkLeNet5(Benchmark):
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
        super(BenchmarkLeNet5, self).__init__(
            epochs, batch_size, n_workers, root, f_lr, b_lr, samples
        )

        train_transform = valid_transform = ToTensor() 

        self.train_loader = DataLoader(
            MNIST(self.root, True, transform=train_transform, download=True),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers,
            pin_memory=True
        )
        self.valid_loader = DataLoader(
            MNIST(self.root, False, transform=valid_transform, download=True),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers,
            pin_memory=True
        )

        self.frequentist = LeNet5(n_classes=10)
        self.bayesian = BayesianLeNet5(n_classes=10)