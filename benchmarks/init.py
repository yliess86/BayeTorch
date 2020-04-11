from bayetorch.metrics import ELBO
from benchmarks.base import step_bayesian
from benchmarks.base import step_frequentist
from typing import Tuple
from torch.optim import Adam
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


class BenchmarkInit:
    def __init__(
        self,
        f_epochs: int, b_epochs: int, f_lr: float, b_lr: float, samples: int,
        train_loader: DataLoader, valid_loader: DataLoader,
        n_classes: int, frequentist: nn.Module, bayesian: nn.Module,
        freeze: bool
    ) -> None:
        self.f_epochs = f_epochs
        self.b_epochs = b_epochs
        self.f_lr = f_lr
        self.b_lr = b_lr
        self.samples = samples
        self.freeze = freeze

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.frequentist = frequentist(n_classes=n_classes)
        self.bayesian = bayesian(n_classes=n_classes)

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