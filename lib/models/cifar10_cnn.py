from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, conv1x1


class ResNet(nn.Module):
    def __init__(self, n: int = 3):
        super().__init__()

        self.n = n

        self.conv1 = nn.Conv2d(3, 16, (3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = self._make_stack(inplanes=16, planes=16)
        self.conv3 = self._make_stack(inplanes=16, planes=32, stride=2)
        self.conv4 = self._make_stack(inplanes=32, planes=64, stride=2)

        self.out_features = 64

    def _make_stack(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
    ) -> nn.Sequential:

        if stride != 1:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )
        else:
            downsample = None

        stack = [
            BasicBlock(
                inplanes=inplanes, planes=planes, stride=stride, downsample=downsample
            )
        ]
        for _ in range(self.n - 1):
            stack.append(BasicBlock(inplanes=planes, planes=planes))
        return nn.Sequential(*stack)

    def _make_downsample(self, in_channels: int, out_channels: int) -> nn.Conv2d:
        return nn.Conv2d(in_channels, out_channels, (3, 3), stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.mean(dim=(-2, -1))
        return x


class CIFAR10Backbone(nn.Module):
    def __init__(self, in_channels: int = 3, out_features: int = 84, *args, **kwargs):
        super().__init__()

        self.out_features = out_features

        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, out_features)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
