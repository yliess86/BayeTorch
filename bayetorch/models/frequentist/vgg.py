from torch import Tensor

import torch
import torch.nn as nn


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
        
        self.fc1 = nn.Linear(512,       512)
        self.fc2 = nn.Linear(512, n_classes)

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
        X = self.fc2(X)
        
        return X


class VGG13(nn.Module):
    def __init__(self, n_classes: int = 10) -> None:
        super(VGG13, self).__init__()
        self.conv1  = nn.Conv2d(  3,  64, 3, padding=1)
        self.conv2  = nn.Conv2d( 64,  64, 3, padding=1)
        self.conv3  = nn.Conv2d( 64, 128, 3, padding=1)
        self.conv4  = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5  = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6  = nn.Conv2d(256, 256, 3, padding=1)
        self.conv7  = nn.Conv2d(256, 512, 3, padding=1)
        self.conv8  = nn.Conv2d(512, 512, 3, padding=1)
        self.conv9  = nn.Conv2d(512, 512, 3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)
        
        self.fc1 = nn.Linear(512,       512)
        self.fc2 = nn.Linear(512, n_classes)

    def forward(self, X: Tensor) -> Tensor:
        X = torch.relu(self.conv1(X))
        X = torch.max_pool2d(torch.relu(self.conv2(X)), 2, stride=2)
        X = torch.relu(self.conv3(X))
        X = torch.max_pool2d(torch.relu(self.conv4(X)), 2, stride=2)
        X = torch.relu(self.conv5(X))
        X = torch.max_pool2d(torch.relu(self.conv6(X)), 2, stride=2)
        X = torch.relu(self.conv7(X))
        X = torch.max_pool2d(torch.relu(self.conv8(X)), 2, stride=2)
        X = torch.relu(self.conv9(X))
        X = torch.max_pool2d(torch.relu(self.conv10(X)), 2, stride=2)
        
        X = X.view(X.size(0), -1)
        
        X = torch.relu(self.fc1(X))
        X = self.fc2(X)

        return X


class VGG16(nn.Module):
    def __init__(self, n_classes: int = 10) -> None:
        super(VGG16, self).__init__()
        self.conv1  = nn.Conv2d(  3,  64, 3, padding=1)
        self.conv2  = nn.Conv2d( 64,  64, 3, padding=1)
        self.conv3  = nn.Conv2d( 64, 128, 3, padding=1)
        self.conv4  = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5  = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6  = nn.Conv2d(256, 256, 3, padding=1)
        self.conv7  = nn.Conv2d(256, 256, 3, padding=1)
        self.conv8  = nn.Conv2d(256, 512, 3, padding=1)
        self.conv9  = nn.Conv2d(512, 512, 3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv11 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv13 = nn.Conv2d(512, 512, 3, padding=1)
        
        self.fc1 = nn.Linear(512,       512)
        self.fc2 = nn.Linear(512, n_classes)

    def forward(self, X: Tensor) -> Tensor:
        X = torch.relu(self.conv1(X))
        X = torch.max_pool2d(torch.relu(self.conv2(X)), 2, stride=2)
        X = torch.relu(self.conv3(X))
        X = torch.max_pool2d(torch.relu(self.conv4(X)), 2, stride=2)
        X = torch.relu(self.conv5(X))
        X = torch.relu(self.conv6(X))
        X = torch.max_pool2d(torch.relu(self.conv7(X)), 2, stride=2)
        X = torch.relu(self.conv8(X))
        X = torch.relu(self.conv9(X))
        X = torch.max_pool2d(torch.relu(self.conv10(X)), 2, stride=2)
        X = torch.relu(self.conv11(X))
        X = torch.relu(self.conv12(X))
        X = torch.max_pool2d(torch.relu(self.conv13(X)), 2, stride=2)
        
        X = X.view(X.size(0), -1)
        
        X = torch.relu(self.fc1(X))
        X = self.fc2(X)

        return X


class VGG19(nn.Module):
    def __init__(self, n_classes: int = 10) -> None:
        super(VGG19, self).__init__()
        self.conv1  = nn.Conv2d(  3,  64, 3, padding=1)
        self.conv2  = nn.Conv2d( 64,  64, 3, padding=1)
        self.conv3  = nn.Conv2d( 64, 128, 3, padding=1)
        self.conv4  = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5  = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6  = nn.Conv2d(256, 256, 3, padding=1)
        self.conv7  = nn.Conv2d(256, 256, 3, padding=1)
        self.conv8  = nn.Conv2d(256, 256, 3, padding=1)
        self.conv9  = nn.Conv2d(256, 512, 3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv11 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv13 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv14 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv15 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv16 = nn.Conv2d(512, 512, 3, padding=1)
        
        self.fc1 = nn.Linear(512,       512)
        self.fc2 = nn.Linear(512, n_classes)

    def forward(self, X: Tensor) -> Tensor:
        X = torch.relu(self.conv1(X))
        X = torch.max_pool2d(torch.relu(self.conv2(X)), 2, stride=2)
        X = torch.relu(self.conv3(X))
        X = torch.max_pool2d(torch.relu(self.conv4(X)), 2, stride=2)
        X = torch.relu(self.conv5(X))
        X = torch.relu(self.conv6(X))
        X = torch.relu(self.conv7(X))
        X = torch.max_pool2d(torch.relu(self.conv8(X)), 2, stride=2)
        X = torch.relu(self.conv9(X))
        X = torch.relu(self.conv10(X))
        X = torch.relu(self.conv11(X))
        X = torch.max_pool2d(torch.relu(self.conv12(X)), 2, stride=2)
        X = torch.relu(self.conv13(X))
        X = torch.relu(self.conv14(X))
        X = torch.relu(self.conv15(X))
        X = torch.max_pool2d(torch.relu(self.conv16(X)), 2, stride=2)
        
        X = X.view(X.size(0), -1)
        
        X = torch.relu(self.fc1(X))
        X = self.fc2(X)

        return X