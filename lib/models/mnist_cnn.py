import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


class MNISTBackbone(nn.Module):
    def __init__(
        self, in_channels: int = 1, out_features: int = 10, spectral_norm: bool = False, leaky_relu: bool = False
    ):
        super().__init__()

        self.out_features = out_features
        self.spectral_norm = spectral_norm
        self.leaky_relu = leaky_relu

        self.conv_1 = self._spectral_norm(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        self.conv_2 = self._spectral_norm(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            )
        )
        self.fc_1 = self._spectral_norm(
            nn.Linear(in_features=7 * 7 * 64, out_features=524)
        )
        self.fc_2 = self._spectral_norm(
            nn.Linear(in_features=524, out_features=out_features)
        )

        self.relu = nn.LeakyReLU() if leaky_relu else nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv_1(x))
        x = self.pool(x)
        x = self.relu(self.conv_2(x))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x

    def _spectral_norm(self, layer: nn.Module) -> nn.Module:
        if self.spectral_norm:
            layer = spectral_norm(layer)
        return layer
