from bayetorch.layers.conv2d import BayesianConv2D
from bayetorch.layers.linear import BayesianLinear
from bayetorch.models.base import BayesianModel
from torch import Tensor
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianVGG11(BayesianModel):
    def __init__(self, n_classes: int = 10) -> None:
        super(BayesianVGG11, self).__init__()
        self.n_classes = n_classes

        self.conv1 = BayesianConv2D(  3,  64, 3, padding=1)
        self.conv2 = BayesianConv2D( 64, 128, 3, padding=1)
        self.conv3 = BayesianConv2D(128, 256, 3, padding=1)
        self.conv4 = BayesianConv2D(256, 256, 3, padding=1)
        self.conv5 = BayesianConv2D(256, 512, 3, padding=1)
        self.conv6 = BayesianConv2D(512, 512, 3, padding=1)
        self.conv7 = BayesianConv2D(512, 512, 3, padding=1)
        self.conv8 = BayesianConv2D(512, 512, 3, padding=1)
        
        self.fc1 = BayesianLinear(512,       512)
        self.fc2 = BayesianLinear(512, n_classes)

    def forward(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        X = torch.max_pool2d(F.softplus(self.conv1(X)), 2, stride=2)
        X = torch.max_pool2d(F.softplus(self.conv2(X)), 2, stride=2)
        X = F.softplus(self.conv3(X))
        X = torch.max_pool2d(F.softplus(self.conv4(X)), 2, stride=2)
        X = F.softplus(self.conv5(X))
        X = torch.max_pool2d(F.softplus(self.conv6(X)), 2, stride=2)
        X = F.softplus(self.conv7(X))
        X = torch.max_pool2d(F.softplus(self.conv8(X)), 2, stride=2)
        
        X = X.view(X.size(0), -1)
        
        X = F.softplus(self.fc1(X))
        X = self.fc2(X)
        
        kld = self.kl_divergence
        
        return X, kld


class BayesianVGG13(BayesianModel):
    def __init__(self, n_classes: int = 10) -> None:
        super(BayesianVGG13, self).__init__()
        self.n_classes = n_classes
        
        self.conv1  = BayesianConv2D(  3,  64, 3, padding=1)
        self.conv2  = BayesianConv2D( 64,  64, 3, padding=1)
        self.conv3  = BayesianConv2D( 64, 128, 3, padding=1)
        self.conv4  = BayesianConv2D(128, 128, 3, padding=1)
        self.conv5  = BayesianConv2D(128, 256, 3, padding=1)
        self.conv6  = BayesianConv2D(256, 256, 3, padding=1)
        self.conv7  = BayesianConv2D(256, 512, 3, padding=1)
        self.conv8  = BayesianConv2D(512, 512, 3, padding=1)
        self.conv9  = BayesianConv2D(512, 512, 3, padding=1)
        self.conv10 = BayesianConv2D(512, 512, 3, padding=1)
        
        self.fc1 = BayesianLinear(512,       512)
        self.fc2 = BayesianLinear(512, n_classes)

    def forward(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        X = F.softplus(self.conv1(X))
        X = torch.max_pool2d(F.softplus(self.conv2(X)), 2, stride=2)
        X = F.softplus(self.conv3(X))
        X = torch.max_pool2d(F.softplus(self.conv4(X)), 2, stride=2)
        X = F.softplus(self.conv5(X))
        X = torch.max_pool2d(F.softplus(self.conv6(X)), 2, stride=2)
        X = F.softplus(self.conv7(X))
        X = torch.max_pool2d(F.softplus(self.conv8(X)), 2, stride=2)
        X = F.softplus(self.conv9(X))
        X = torch.max_pool2d(F.softplus(self.conv10(X)), 2, stride=2)
        
        X = X.view(X.size(0), -1)
        
        X = F.softplus(self.fc1(X))
        X = self.fc2(X)

        kld = self.kl_divergence
        
        return X, kld


class BayesianVGG16(BayesianModel):
    def __init__(self, n_classes: int = 10) -> None:
        super(BayesianVGG16, self).__init__()
        self.n_classes = n_classes
        
        self.conv1  = BayesianConv2D(  3,  64, 3, padding=1)
        self.conv2  = BayesianConv2D( 64,  64, 3, padding=1)
        self.conv3  = BayesianConv2D( 64, 128, 3, padding=1)
        self.conv4  = BayesianConv2D(128, 128, 3, padding=1)
        self.conv5  = BayesianConv2D(128, 256, 3, padding=1)
        self.conv6  = BayesianConv2D(256, 256, 3, padding=1)
        self.conv7  = BayesianConv2D(256, 256, 3, padding=1)
        self.conv8  = BayesianConv2D(256, 512, 3, padding=1)
        self.conv9  = BayesianConv2D(512, 512, 3, padding=1)
        self.conv10 = BayesianConv2D(512, 512, 3, padding=1)
        self.conv11 = BayesianConv2D(512, 512, 3, padding=1)
        self.conv12 = BayesianConv2D(512, 512, 3, padding=1)
        self.conv13 = BayesianConv2D(512, 512, 3, padding=1)
        
        self.fc1 = BayesianLinear(512,       512)
        self.fc2 = BayesianLinear(512, n_classes)

    def forward(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        X = F.softplus(self.conv1(X))
        X = torch.max_pool2d(F.softplus(self.conv2(X)), 2, stride=2)
        X = F.softplus(self.conv3(X))
        X = torch.max_pool2d(F.softplus(self.conv4(X)), 2, stride=2)
        X = F.softplus(self.conv5(X))
        X = F.softplus(self.conv6(X))
        X = torch.max_pool2d(F.softplus(self.conv7(X)), 2, stride=2)
        X = F.softplus(self.conv8(X))
        X = F.softplus(self.conv9(X))
        X = torch.max_pool2d(F.softplus(self.conv10(X)), 2, stride=2)
        X = F.softplus(self.conv11(X))
        X = F.softplus(self.conv12(X))
        X = torch.max_pool2d(F.softplus(self.conv13(X)), 2, stride=2)
        
        X = X.view(X.size(0), -1)
        
        X = F.softplus(self.fc1(X))
        X = self.fc2(X)

        kld = self.kl_divergence
        
        return X, kld


class BayesianVGG19(BayesianModel):
    def __init__(self, n_classes: int = 10) -> None:
        super(BayesianVGG19, self).__init__()
        self.n_classes = n_classes
        
        self.conv1  = BayesianConv2D(  3,  64, 3, padding=1)
        self.conv2  = BayesianConv2D( 64,  64, 3, padding=1)
        self.conv3  = BayesianConv2D( 64, 128, 3, padding=1)
        self.conv4  = BayesianConv2D(128, 128, 3, padding=1)
        self.conv5  = BayesianConv2D(128, 256, 3, padding=1)
        self.conv6  = BayesianConv2D(256, 256, 3, padding=1)
        self.conv7  = BayesianConv2D(256, 256, 3, padding=1)
        self.conv8  = BayesianConv2D(256, 256, 3, padding=1)
        self.conv9  = BayesianConv2D(256, 512, 3, padding=1)
        self.conv10 = BayesianConv2D(512, 512, 3, padding=1)
        self.conv11 = BayesianConv2D(512, 512, 3, padding=1)
        self.conv12 = BayesianConv2D(512, 512, 3, padding=1)
        self.conv13 = BayesianConv2D(512, 512, 3, padding=1)
        self.conv14 = BayesianConv2D(512, 512, 3, padding=1)
        self.conv15 = BayesianConv2D(512, 512, 3, padding=1)
        self.conv16 = BayesianConv2D(512, 512, 3, padding=1)
        
        self.fc1 = BayesianLinear(512,       512)
        self.fc2 = BayesianLinear(512, n_classes)

    def forward(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        X = F.softplus(self.conv1(X))
        X = torch.max_pool2d(F.softplus(self.conv2(X)), 2, stride=2)
        X = F.softplus(self.conv3(X))
        X = torch.max_pool2d(F.softplus(self.conv4(X)), 2, stride=2)
        X = F.softplus(self.conv5(X))
        X = F.softplus(self.conv6(X))
        X = F.softplus(self.conv7(X))
        X = torch.max_pool2d(F.softplus(self.conv8(X)), 2, stride=2)
        X = F.softplus(self.conv9(X))
        X = F.softplus(self.conv10(X))
        X = F.softplus(self.conv11(X))
        X = torch.max_pool2d(F.softplus(self.conv12(X)), 2, stride=2)
        X = F.softplus(self.conv13(X))
        X = F.softplus(self.conv14(X))
        X = F.softplus(self.conv15(X))
        X = torch.max_pool2d(F.softplus(self.conv16(X)), 2, stride=2)
        
        X = X.view(X.size(0), -1)
        
        X = F.softplus(self.fc1(X))
        X = self.fc2(X)

        kld = self.kl_divergence
        
        return X, kld