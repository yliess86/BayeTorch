from bayetorch.metrics import ELBO
from benchmarks.helpers import step_bayesian
from benchmarks.helpers import step_frequentist
from typing import List
from typing import Tuple
from torch import Tensor
from torch.optim import Adam
from torch.optim import Optimizer
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Benchmark:
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
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.root = root
        self.f_lr = f_lr
        self.b_lr = b_lr
        self.samples = samples

        self.train_loader = None
        self.valid_loader = None

        self.frequentist = None
        self.bayesian = None

    def run_frequentist(self) -> Tuple[float, float]:
        print("---- Frequentist")
        
        megabytes, _ = SizeEstimator(self.frequentist, (1, 1, 28, 28))()

        self.frequentist = self.frequentist.cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optim = Adam(self.frequentist.parameters(), lr=self.f_lr)

        best_acc = 0.0
        for epoch in range(self.epochs):
            print("-- Epoch", f"{(epoch + 1)}/{self.epochs}")

            self.frequentist.train()
            n = len(self.train_loader.dataset)
            for X, Y in tqdm(self.train_loader, desc="Train Batch"):
                step_frequentist(
                    self.frequentist, X, Y, n, optim, criterion, False
                )

            self.frequentist.eval()
            n = len(self.valid_loader.dataset)
            acc = 0.0
            for X, Y in tqdm(self.valid_loader, desc="Validation Batch"):
                loss, acc_ = step_frequentist(
                    self.frequentist, X, Y, n, optim, criterion, True
                )
                acc += acc_
            
            print("Validation Accuracy:", f"{acc:.2%}")
            if acc > best_acc:
                best_acc = acc

        print()

        return best_acc, megabytes

    def run_bayesian(self) -> Tuple[float, float]:
        print("---- Bayesian")

        megabytes, _ = SizeEstimator(self.bayesian, (1, 1, 28, 28))()

        self.bayesian = self.bayesian.cuda()
        criterion = ELBO(len(self.train_loader.dataset)).cuda()
        optim = Adam(self.bayesian.parameters(), lr=self.b_lr)

        best_acc = 0.0
        for epoch in range(self.epochs):
            print("-- Epoch", f"{(epoch + 1)}/{self.epochs}")

            self.bayesian.train()
            n = len(self.train_loader.dataset)
            for X, Y in tqdm(self.train_loader, desc="Train Batch"):
                step_bayesian(
                    self.bayesian, X, Y, n, optim, criterion, 
                    self.samples, False
                )

            self.bayesian.eval()
            n = len(self.valid_loader.dataset)
            acc = 0.0
            for X, Y in tqdm(self.valid_loader, desc="Validation Batch"):
                loss, acc_ = step_bayesian(
                    self.bayesian, X, Y, n, optim, criterion,
                    self.samples, True
                )
                acc += acc_
            
            print("Validation Accuracy:", f"{acc:.2%}")
            if acc > best_acc:
                best_acc = acc

        print()

        return best_acc, megabytes

    def __call__(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        frequentist = self.run_frequentist()
        bayesian = self.run_bayesian()

        return frequentist, bayesian


class BenchmarkWithInit:
    def __init__(
        self,
        f_epochs: int,
        b_epochs: int,
        batch_size: int,
        n_workers: int,
        root: str,
        f_lr: float,
        b_lr: float,
        freeze: bool,
        samples: int
    ) -> None:
        super(BenchmarkWithInit, self).__init__()
        self.f_epochs = f_epochs
        self.b_epochs = b_epochs
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.root = root
        self.f_lr = f_lr
        self.b_lr = b_lr
        self.freeze = freeze
        self.samples = samples

        self.train_loader = None
        self.valid_loader = None

        self.frequentist = None
        self.bayesian = None

    def pretrain(self) -> float:
        print("---- Pretrain")
        self.frequentist = self.frequentist.cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optim = Adam(self.frequentist.parameters(), lr=self.f_lr)

        best_acc = 0.0
        for epoch in range(self.f_epochs):
            print("-- Epoch", f"{(epoch + 1)}/{self.f_epochs}")

            self.frequentist.train()
            n = len(self.train_loader.dataset)
            for X, Y in tqdm(self.train_loader, desc="Train Batch"):
                step_frequentist(
                    self.frequentist, X, Y, n, optim, criterion, False
                )

            self.frequentist.eval()
            n = len(self.valid_loader.dataset)
            acc = 0.0
            for X, Y in tqdm(self.valid_loader, desc="Validation Batch"):
                loss, acc_ = step_frequentist(
                    self.frequentist, X, Y, n, optim, criterion, True
                )
                acc += acc_
            
            print("Validation Accuracy:", f"{acc:.2%}")
            if acc > best_acc:
                best_acc = acc

        print()
        self.frequentist = self.frequentist.cpu()

        return best_acc

    def finetune(self) -> Tuple[float, float]:
        print("---- Bayesian")

        self.bayesian = self.bayesian.init_with_frequentist(
            self.frequentist, self.freeze
        )

        self.bayesian = self.bayesian.cuda()
        criterion = ELBO(len(self.train_loader.dataset)).cuda()
        optim = Adam(self.bayesian.parameters(), lr=self.b_lr)

        self.bayesian.eval()
        n = len(self.valid_loader.dataset)
        start_acc = 0.0
        for X, Y in tqdm(self.valid_loader, desc="Validation Batch"):
            loss, acc_ = step_bayesian(
                self.bayesian, X, Y, n, optim, criterion,
                self.samples, True
            )
            start_acc += acc_
        
        print("Start Validation Accuracy:", f"{start_acc:.2%}")

        best_acc = 0.0
        for epoch in range(self.b_epochs):
            print("-- Epoch", f"{(epoch + 1)}/{self.b_epochs}")

            self.bayesian.train()
            n = len(self.train_loader.dataset)
            for X, Y in tqdm(self.train_loader, desc="Train Batch"):
                step_bayesian(
                    self.bayesian, X, Y, n, optim, criterion, 
                    self.samples, False
                )

            self.bayesian.eval()
            n = len(self.valid_loader.dataset)
            acc = 0.0
            for X, Y in tqdm(self.valid_loader, desc="Validation Batch"):
                loss, acc_ = step_bayesian(
                    self.bayesian, X, Y, n, optim, criterion,
                    self.samples, True
                )
                acc += acc_
            
            print("Validation Accuracy:", f"{acc:.2%}")
            if acc > best_acc:
                best_acc = acc

        print()

        return start_acc, best_acc

    def __call__(self) -> Tuple[float, float, float]:
        pretrained_acc = self.pretrain()
        start_acc, acc = self.finetune()

        return pretrained_acc, start_acc, acc


class SizeEstimator(object):
    def __init__(
        self,
        model: nn.Module,
        input_size: List[int],
        bit_size: int = 32
    ) -> None:
        self.model = model
        self.input_size = input_size
        self.bit_size = 32

    @property
    def parameter_sizes(self) -> List[np.ndarray]:
        return [
            np.array(parameter.size())
            for module in self.model.modules()
            for parameter in module.parameters()
        ]

    @property
    def param_bits(self) -> int:
        parameter_sizes = self.parameter_sizes
        bits = np.sum([
            np.prod(size) for size in parameter_sizes]
        ) * self.bit_size
        
        return bits

    def __call__(self) -> Tuple[float, int]:
        bits = self.param_bits
        megabytes = (bits / 8) / (1024 ** 2)
        
        return megabytes, bits