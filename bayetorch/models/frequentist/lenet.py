from torch import Tensor

import torch
import torch.nn as nn


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