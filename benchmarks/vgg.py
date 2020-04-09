import os
import sys

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, ".."))

from bayetorch.metrics import ELBO
from bayetorch.models import BayesianVGG11
from benchmarks.base import Benchmark
from benchmarks.base import SizeEstimator
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from tqdm import tqdm
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG11(nn.Module):
    def __init__(self, n_classes: int = 10) -> None:
        super(VGG11, self).__init__()
        self.conv1 = nn.Conv2d(  3,  64, 3, padding=1)
        self.conv2 = nn.Conv2d( 64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv7 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        
        self.fc1 = nn.Linear(512 * 7 * 7, 512)
        self.fc2 = nn.Linear(512,         512)
        self.fc3 = nn.Linear(512,   n_classes)

    def forward(self, X: Tensor) -> Tensor:
        X = torch.max_pool2d(torch.relu(self.conv1(X)), 2, stride=2)
        X = torch.max_pool2d(torch.relu(self.conv2(X)), 2, stride=2)
        X = torch.relu(self.conv3(X))
        X = torch.max_pool2d(torch.relu(self.conv4(X)), 2, stride=2)
        X = torch.relu(self.conv5(X))
        X = torch.max_pool2d(torch.relu(self.conv6(X)), 2, stride=2)
        X = torch.relu(self.conv7(X))
        X = torch.max_pool2d(torch.relu(self.conv8(X)), 2, stride=2)
        
        X = X.view(X.size(0), -1)
        
        X = torch.relu(self.fc1(X))
        X = torch.relu(self.fc2(X))
        X = self.fc3(X)
        
        return X


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

        train_transform = valid_transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

        print()

        return acc, megabytes

    def run_bayesian(self) -> Tuple[float, float]:
        print("---- Bayesian")

        model = BayesianVGG11(n_classes=10)
        megabytes, _ = SizeEstimator(model, (1, 3, 224, 224))()

        model = model.cuda()
        criterion = ELBO(len(self.train_loader.dataset)).cuda()
        optim = Adam(model.parameters(), lr=self.b_lr)

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

        print()

        return acc, megabytes