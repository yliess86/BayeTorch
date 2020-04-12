import sys
import os

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, ".."))

from bayetorch.models import BayesianLeNet5
from bayetorch.models import BayesianVGG11
from bayetorch.models import BayesianVGG16
from bayetorch.models import LeNet5
from bayetorch.models import VGG11
from bayetorch.models import VGG16
from benchmarks.acc import BenchmarkAcc
from benchmarks.init import BenchmarkInit
from benchmarks.uncert import BenchmarkUncert
from prettytable import PrettyTable
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torchvision.transforms import RandomCrop
from torchvision.transforms import RandomHorizontalFlip
from typing import Tuple


import argparse


def cifar10(
    batch_size: int, n_workers: int, root: str = "datasets"
) -> Tuple[DataLoader, DataLoader]:
    normalize = Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )

    valid_t = Compose([ToTensor(), normalize])
    train_t = Compose([RandomCrop(32, 4), RandomHorizontalFlip(), valid_t])

    train_loader = DataLoader(
        CIFAR10(root, True, transform=train_t, download=True),
        batch_size=batch_size, shuffle=True, num_workers=n_workers,
        pin_memory=True
    )
    valid_loader = DataLoader(
        CIFAR10(root, False, transform=valid_t, download=True),
        batch_size=batch_size, shuffle=False, num_workers=n_workers,
        pin_memory=True
    )

    return train_loader, valid_loader


def mnist(
    batch_size: int, n_workers: int, root: str = "datasets"
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        MNIST(root, True, transform=ToTensor(), download=True),
        batch_size=batch_size, shuffle=True, num_workers=n_workers,
        pin_memory=True
    )
    valid_loader = DataLoader(
        MNIST(root, False, transform=ToTensor(), download=True),
        batch_size=batch_size, shuffle=False, num_workers=n_workers,
        pin_memory=True
    )

    return train_loader, valid_loader


def undefined_model() -> None:
    print("The specified model does not exists or is not available yet.")
    exit(1)


def undefined_task() -> None:
    print("The specified task does not exists or is not available yet.")
    exit(1)


def benchmark_acc(
    name: str, freq, baye, train, valid, n_classes, args
) -> None:
    (f_acc, f_size), (b_acc, b_size) = BenchmarkAcc(
        *args.epochs, *args.lr, args.samples,
        train, valid, n_classes, freq, baye
    )()

    table = PrettyTable([
        "Model", 
        "Frequentist Acc",  "Bayesian Acc",
        "Frequentist Size", "Bayesian Size"
    ])
    table.add_row([
        name, 
        f"{f_acc:.2%}",   f"{b_acc:.2%}",
        f"{f_size:.4}Mb", f"{b_size:.4}Mb"
    ])

    print("---- Table")
    print(table)
    exit(0)


def benchmark_init(
    name: str, freq, baye, train, valid, n_classes, args
) -> None:
    p_acc, s_acc, f_acc = BenchmarkInit(
        *args.epochs, *args.lr, args.samples,
        train, valid, n_classes, freq, baye, args.freeze
    )()
    
    table = PrettyTable([
        "Model", "Pretrained Acc", "Starting Acc", "Final Acc"
    ])
    table.add_row([name, f"{p_acc:.2%}", f"{s_acc:.2%}", f"{f_acc:.2%}"])

    print("---- Table")
    print(table)
    exit(0)


def benchmark_uncert(
    name: str, freq, baye, train, valid, n_classes, args
) -> None:
    BenchmarkUncert(
        *args.epochs, *args.lr, args.samples,
        train, valid, n_classes, baye
    )()

    exit(0)


CHOICES = {
    "acc":    benchmark_acc,
    "init":   benchmark_init,
    "uncert": benchmark_uncert
}

parser = argparse.ArgumentParser()
parser.add_argument("--model",      type=str,        required=True)
parser.add_argument("--epochs",     type=int,        required=True, nargs="+")
parser.add_argument("--batch_size", type=int,        required=True)
parser.add_argument("--n_workers",  type=int,        default=6)
parser.add_argument("--lr",         type=float,      required=True, nargs="+")
parser.add_argument("--samples",    type=int,        default=1)
parser.add_argument("--task",       type=str,        required=True)
parser.add_argument("--freeze",     action="store_true")

args = parser.parse_args()

params = (args.batch_size, args.n_workers)
BENCHMARKS = {
    "LeNet": [LeNet5, BayesianLeNet5, mnist(*params),   10],
    "VGG11": [VGG11,  BayesianVGG11,  cifar10(*params), 10],
    "VGG16": [VGG16,  BayesianVGG16,  cifar10(*params), 10]
}

name = None
freq, baye, dataset, n_classes = None, None, None, 0
for model_name in BENCHMARKS.keys():
    if args.model.lower() == model_name.lower():
        name = model_name
        freq, baye, dataset, n_classes = BENCHMARKS[name]

if name is None:
    undefined_model()

task = None
for choice in CHOICES.keys():
    if args.task.lower() == choice.lower():
        print(f"\n====== {choice.upper()} ======\n")
        CHOICES[choice](name, freq, baye, *dataset, n_classes, args)

if task is None:
    undefined_task()