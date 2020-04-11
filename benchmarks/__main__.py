from benchmarks.lenet import BenchmarkLeNet5
from benchmarks.vgg import BenchmarkVGG11
from benchmarks.vgg import BenchmarkVGG16
from benchmarks.lenet_with_init import BenchmarkWithInitLeNet5
from benchmarks.vgg_with_init import BenchmarkWithInitVGG11
from benchmarks.vgg_with_init import BenchmarkWithInitVGG16
from prettytable import PrettyTable

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model",      type=str,   required=True)
parser.add_argument("--epochs",     type=int,   required=True, nargs="+")
parser.add_argument("--batch_size", type=int,   required=True)
parser.add_argument("--lr",         type=float, required=True, nargs="+")
parser.add_argument("--init",       action="store_true")
parser.add_argument("--freeze",     action="store_true")

args = parser.parse_args()

if args.init:
    if args.model.lower() == "LeNet".lower():
        p_acc, s_acc, f_acc = BenchmarkWithInitLeNet5(
            f_epochs=args.epochs[0], b_epochs=args.epochs[1],
            batch_size=args.batch_size, n_workers=6, root="datasets",
            f_lr=args.lr[0], b_lr=args.lr[1], freeze=args.freeze
        )()
        
        table = PrettyTable([
            "Model", "Pretrained Acc",  "Starting Acc", "Final Acc"
        ])
        table.add_row([
            "LeNet5", f"{p_acc:.2%}", f"{s_acc:.2%}", f"{f_acc:.2%}"
        ])

        print("---- Table")
        print(table)
        exit(0)

    if args.model.lower() == "VGG11".lower():
        p_acc, s_acc, f_acc = BenchmarkWithInitVGG11(
            f_epochs=args.epochs[0], b_epochs=args.epochs[1],
            batch_size=args.batch_size, n_workers=6, root="datasets",
            f_lr=args.lr[0], b_lr=args.lr[1], freeze=args.freeze
        )()
        
        table = PrettyTable([
            "Model", "Pretrained Acc",  "Starting Acc", "Final Acc"
        ])
        table.add_row([
            "VGG11", f"{p_acc:.2%}", f"{s_acc:.2%}", f"{f_acc:.2%}"
        ])

        print("---- Table")
        print(table)
        exit(0)

    if args.model.lower() == "VGG16".lower():
        p_acc, s_acc, f_acc = BenchmarkWithInitVGG16(
            f_epochs=args.epochs[0], b_epochs=args.epochs[1],
            batch_size=args.batch_size, n_workers=6, root="datasets",
            f_lr=args.lr[0], b_lr=args.lr[1], freeze=args.freeze
        )()
        
        table = PrettyTable([
            "Model", "Pretrained Acc",  "Starting Acc", "Final Acc"
        ])
        table.add_row([
            "VGG16", f"{p_acc:.2%}", f"{s_acc:.2%}", f"{f_acc:.2%}"
        ])

        print("---- Table")
        print(table)
        exit(0)

else:
    if args.model.lower() == "LeNet".lower():
        (f_acc, f_size), (b_acc, b_size) = BenchmarkLeNet5(
            epochs=args.epochs[0], batch_size=args.batch_size, 
            n_workers=6, root="datasets", f_lr=args.lr[0], b_lr=args.lr[1],
        )()

        table = PrettyTable([
            "Model", 
            "Frequentist Acc",  "Bayesian Acc",
            "Frequentist Size", "Bayesian Size"
        ])
        table.add_row([
            "LeNet5", 
            f"{f_acc:.2%}",   f"{b_acc:.2%}",
            f"{f_size:.4}Mb", f"{b_size:.4}Mb"
        ])

        print("---- Table")
        print(table)
        exit(0)

    if args.model.lower() == "VGG11".lower():
        (f_acc, f_size), (b_acc, b_size) = BenchmarkVGG11(
            epochs=args.epochs[0], batch_size=args.batch_size,
            n_workers=6, root="datasets", f_lr=args.lr[0], b_lr=args.lr[1],
        )()

        table = PrettyTable([
            "Model", 
            "Frequentist Acc",  "Bayesian Acc",
            "Frequentist Size", "Bayesian Size"
        ])
        table.add_row([
            "VGG11", 
            f"{f_acc:.2%}",   f"{b_acc:.2%}",
            f"{f_size:.4}Mb", f"{b_size:.4}Mb"
        ])

        print("---- Table")
        print(table)
        exit(0)

    if args.model.lower() == "VGG16".lower():
        (f_acc, f_size), (b_acc, b_size) = BenchmarkVGG16(
            epochs=args.epochs[0], batch_size=args.batch_size,
            n_workers=6, root="datasets", f_lr=args.lr[0], b_lr=args.lr[1],
        )()

        table = PrettyTable([
            "Model", 
            "Frequentist Acc",  "Bayesian Acc",
            "Frequentist Size", "Bayesian Size"
        ])
        table.add_row([
            "VGG11", 
            f"{f_acc:.2%}",   f"{b_acc:.2%}",
            f"{f_size:.4}Mb", f"{b_size:.4}Mb"
        ])

        print("---- Table")
        print(table)
        exit(0)

print("The specified model does not exists or is not available yet.")
exit(1)