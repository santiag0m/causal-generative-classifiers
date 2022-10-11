import torch
import torch.nn as nn


class DotClassifier(nn.Module):
    def __init__(self, backbone: int, num_classes: int = 10, num_layers: int = 1):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.hidden_dim = backbone.out_features

        self.classifier = nn.Sequential(
            *[
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
            ]
            * num_layers,
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        logits_y_z = self.classifier(z)
        return logits_y_z
