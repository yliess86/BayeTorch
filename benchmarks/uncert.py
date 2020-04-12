from bayetorch.metrics import ELBO
from bayetorch.metrics import Uncertainty
from benchmarks.base import step_bayesian
from torch import Tensor
from torch.optim import Adam
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
from torchvision.transforms import RandomRotation
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BenchmarkUncert:
    def __init__(
        self,
        epochs: int, lr: float, samples: int,
        train_loader: DataLoader, valid_loader: DataLoader,
        n_classes: int, bayesian: nn.Module,
    ) -> None:
        self.epochs = epochs
        self.lr = lr
        self.samples = samples

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.bayesian = bayesian(n_classes=n_classes)

    def pretrain(self) -> None:
        print("---- Pretrain")
        
        self.bayesian = self.bayesian.cuda()
        criterion = ELBO(len(self.train_loader.dataset)).cuda()
        optim = Adam(self.bayesian.parameters(), lr=self.lr)

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

    def plot(self) -> None:
        self.bayesian = self.bayesian.eval()
        self.bayesian = self.bayesian.cuda()

        size = 12
        idxs = np.random.randint(0, len(self.valid_loader.dataset), size=size)
        uncertainty = Uncertainty(self.samples)

        img_fig, img_axes = plt.subplots(size, size, figsize=(8, 8))
        
        for j, idx in enumerate(idxs):
            for i in range(size):
                img, label = self.valid_loader.dataset[idx]

                step = (1 / size) * i
                angle = 0 * (1 - step) + 360 * step
                transform = Compose([
                    ToPILImage(),
                    RandomRotation((angle, angle), fill=(0,)),
                    ToTensor() 
                ]) 

                img = transform(img)
                img = img.cuda()

                uncert = uncertainty(self.bayesian, img)
                epsitemic, aleatoric, pred = uncert
                epsitemic = torch.diag(epsitemic).sum().item()
                
                if img.size(0) == 1:
                    img = img.repeat(3, 1, 1)
                img = img.permute(1, 2, 0)
                img = img.detach().cpu().numpy()

                add = np.ones((3, 28, 3))
                add_w = int(np.floor(28 * 0.5 * (epsitemic * 100)))
                add[:, :add_w, :] = [0.2, 0.1, 0.8]

                img = np.vstack([img, add])

                img_ax = img_axes[i, j]
                img_ax.imshow(img)
                img_ax.set_title(f"{label}, {pred}")
                img_ax.set_yticks(())
                img_ax.set_yticklabels(())
                img_ax.set_xticks(())
                img_ax.set_xticklabels(())

        img_fig.tight_layout()
        img_fig.canvas.draw()
        plt.show()

    def __call__(self) -> None:
        self.pretrain()
        self.plot()