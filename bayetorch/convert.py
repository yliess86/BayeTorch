from bayetorch.layers.base import BayesianModule
from bayetorch.models.base import BayesianModel

import torch
import torch.nn as nn


def init_bayesian_with_frequentist(
    bayesian: nn.Module,
    frequentist: nn.Module,
    freeze: bool = True
) -> BayesianModel:
    bayesian_named_params = bayesian.named_parameters()
    frequentist_named_params = frequentist.named_parameters()

    filtr = lambda data: "log_alpha" not in data[0]

    bayesian_named_params = filter(filtr, bayesian_named_params)
    frequentist_named_params = filter(filtr, frequentist_named_params)

    named_params = zip(bayesian_named_params, frequentist_named_params)
    for (_, bayesian_param), (_, frequentist_param) in named_params:
        bayesian_param.data = frequentist_param.data
        if freeze:
            bayesian_param.requires_grad = False


if __name__ == "__main__":
    from bayetorch.models import BayesianLeNet5
    from torch import Tensor

    class LeNet5(nn.Module):
        def __init__(self, n_classes: int = 10) -> None:
            super(LeNet5, self).__init__()
            self.conv1 = nn.Conv2d(1,  6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
            
            self.fc1 = nn.Linear(256,       120)
            self.fc2 = nn.Linear(120,        84)
            self.fc3 = nn.Linear( 84, n_classes)

        def forward(self, X: Tensor) -> Tensor:
            X = torch.max_pool2d(torch.relu(self.conv1(X)), 2)
            X = torch.max_pool2d(torch.relu(self.conv2(X)), 2)
            
            X = X.view(X.size(0), -1)
            
            X = torch.relu(self.fc1(X))
            X = torch.relu(self.fc2(X))
            X = self.fc3(X)
            
            return X


    bayesian_lenet = BayesianLeNet5(n_classes=10)
    frequentist_lenet = LeNet5(n_classes=10)

    init_bayesian_with_frequentist(bayesian_lenet, frequentist_lenet)