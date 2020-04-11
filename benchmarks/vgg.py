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
        b_lr: float
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.root = root
        self.f_lr = f_lr
        self.b_lr = b_lr

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

    def run_frequentist(self) -> Tuple[float, float]:
        print("---- Frequentist")
        
        model = VGG11(n_classes=10)
        megabytes, _ = SizeEstimator(model, (1, 3, 224, 224))()

        model = model.cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optim = Adam(model.parameters(), lr=self.f_lr)

        best_acc = 0.0
        for epoch in range(self.epochs):
            print("-- Epoch", f"{(epoch + 1)}/{self.epochs}")

            model.train()
            for X, Y in tqdm(self.train_loader, desc="Train Batch"):
                optim.zero_grad()
                X, Y = X.float().cuda(), Y.long().cuda()

                y = model(X)
                loss = criterion(y, Y)

                loss.backward()
                optim.step()

            model.eval()
            n = len(self.valid_loader.dataset)
            acc = 0.0
            for X, Y in tqdm(self.valid_loader, desc="Validation Batch"):
                X, Y = X.float().cuda(), Y.long().cuda()

                y = model(X)
                acc += torch.argmax(y, axis=-1).eq(Y).sum().item() / n
            
            print("Validation Accuracy:", f"{acc:.2%}")
            if acc > best_acc:
                best_acc = acc

        print()

        return best_acc, megabytes

    def run_bayesian(self) -> Tuple[float, float]:
        print("---- Bayesian")

        model = BayesianVGG11(n_classes=10)
        megabytes, _ = SizeEstimator(model, (1, 3, 224, 224))()

        model = model.cuda()
        criterion = ELBO(len(self.train_loader.dataset)).cuda()
        optim = Adam(model.parameters(), lr=self.b_lr)

        best_acc = 0.0
        for epoch in range(self.epochs):
            print("-- Epoch", f"{(epoch + 1)}/{self.epochs}")

            model.train()
            for X, Y in tqdm(self.train_loader, desc="Train Batch"):
                optim.zero_grad()
                X, Y = X.float().cuda(), Y.long().cuda()

                y, kld = model(X)
                y = F.log_softmax(y, dim=1)
                log_y = ELBO.log_mean_exp(y.unsqueeze(len(y.shape)), dim=2)
                loss = criterion(log_y, Y, kld)

                loss.backward()
                optim.step()

            model.eval()
            n = len(self.valid_loader.dataset)
            acc = 0.0
            for X, Y in tqdm(self.valid_loader, desc="Validation Batch"):
                X, Y = X.float().cuda(), Y.long().cuda()

                y, _ = model(X)
                y = F.log_softmax(y, dim=1)
                log_y = ELBO.log_mean_exp(y.unsqueeze(len(y.shape)), dim=2)
                acc += torch.argmax(log_y, axis=-1).eq(Y).sum().item() / n
            
            print("Validation Accuracy:", f"{acc:.2%}")
            if acc > best_acc:
                best_acc = acc

        print()

        return best_acc, megabytes


class BenchmarkVGG16(Benchmark):
    def __init__(
        self,
        epochs: int,
        batch_size: int,
        n_workers: int,
        root: str,
        f_lr: float,
        b_lr: float
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.root = root
        self.f_lr = f_lr
        self.b_lr = b_lr

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

    def run_frequentist(self) -> Tuple[float, float]:
        print("---- Frequentist")
        
        model = VGG16(n_classes=10)
        megabytes, _ = SizeEstimator(model, (1, 3, 224, 224))()

        model = model.cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optim = Adam(model.parameters(), lr=self.f_lr)

        best_acc = 0.0
        for epoch in range(self.epochs):
            print("-- Epoch", f"{(epoch + 1)}/{self.epochs}")

            model.train()
            for X, Y in tqdm(self.train_loader, desc="Train Batch"):
                optim.zero_grad()
                X, Y = X.float().cuda(), Y.long().cuda()

                y = model(X)
                loss = criterion(y, Y)

                loss.backward()
                optim.step()

            model.eval()
            n = len(self.valid_loader.dataset)
            acc = 0.0
            for X, Y in tqdm(self.valid_loader, desc="Validation Batch"):
                X, Y = X.float().cuda(), Y.long().cuda()

                y = model(X)
                acc += torch.argmax(y, axis=-1).eq(Y).sum().item() / n
            
            print("Validation Accuracy:", f"{acc:.2%}")
            if acc > best_acc:
                best_acc = acc

        print()

        return best_acc, megabytes

    def run_bayesian(self) -> Tuple[float, float]:
        print("---- Bayesian")

        model = BayesianVGG16(n_classes=10)
        megabytes, _ = SizeEstimator(model, (1, 3, 224, 224))()

        model = model.cuda()
        criterion = ELBO(len(self.train_loader.dataset)).cuda()
        optim = Adam(model.parameters(), lr=self.b_lr)

        best_acc = 0.0
        for epoch in range(self.epochs):
            print("-- Epoch", f"{(epoch + 1)}/{self.epochs}")

            model.train()
            for X, Y in tqdm(self.train_loader, desc="Train Batch"):
                optim.zero_grad()
                X, Y = X.float().cuda(), Y.long().cuda()

                y, kld = model(X)
                y = F.log_softmax(y, dim=1)
                log_y = ELBO.log_mean_exp(y.unsqueeze(len(y.shape)), dim=2)
                loss = criterion(log_y, Y, kld)

                loss.backward()
                optim.step()

            model.eval()
            n = len(self.valid_loader.dataset)
            acc = 0.0
            for X, Y in tqdm(self.valid_loader, desc="Validation Batch"):
                X, Y = X.float().cuda(), Y.long().cuda()

                y, _ = model(X)
                y = F.log_softmax(y, dim=1)
                log_y = ELBO.log_mean_exp(y.unsqueeze(len(y.shape)), dim=2)
                acc += torch.argmax(log_y, axis=-1).eq(Y).sum().item() / n
            
            print("Validation Accuracy:", f"{acc:.2%}")
            if acc > best_acc:
                best_acc = acc

        print()

        return best_acc, megabytes