from typing import List

import torch
import torch.nn as nn


class MLPBackbone(nn.Module):
    def __init__(self, in_features: int, layers: List[int]):
        super().__init__()
        self.out_features = layers[-1]

        fc_layers = []

        for out_features in layers[:-1]:
            fc_layers += [nn.Linear(in_features, out_features), nn.ReLU()]
            in_features = out_features
        fc_layers.append(nn.Linear(in_features, layers[-1]))
        self.mlp = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        x = self.mlp(x)
        return x
