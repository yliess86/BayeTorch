from benchmarks.lenet import BenchmarkLeNet5
from benchmarks.vgg import BenchmarkVGG11
from prettytable import PrettyTable

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)

args = parser.parse_args()

if args.model.lower() == "LeNet".lower():
    (f_acc, f_size), (b_acc, b_size) = BenchmarkLeNet5(
        epochs=2, batch_size=256, n_workers=6, root="datasets", 
        f_lr=1e-2, b_lr=1e-3,
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

if args.model.lower() == "VGG".lower():
    (f_acc, f_size), (b_acc, b_size) = BenchmarkVGG11(
        epochs=1, batch_size=8, n_workers=6, root="datasets", 
        f_lr=1e-2, b_lr=1e-3,
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