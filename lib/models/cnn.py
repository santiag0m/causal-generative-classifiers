import torch
import torch.nn as nn


class CNNBackbone(nn.Module):
    def __init__(self, in_channels: int = 1, out_features: int = 128):
        super().__init__()

        self.out_features = out_features

        self.conv_1 = nn.Conv2d(
            in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv_2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.fc_1 = nn.Linear(in_features=7 * 7 * 64, out_features=self.out_features)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv_1(x))
        x = self.pool(x)
        x = self.relu(self.conv_2(x))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc_1(x))
        return x
