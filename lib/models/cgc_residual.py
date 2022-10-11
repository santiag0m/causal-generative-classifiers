from typing import Union, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils.parametrizations import spectral_norm

from .mnist_cnn import MNISTBackbone
from .mlp import MLPBackbone


class CGCResidual(nn.Module):
    def __init__(
        self,
        backbone: Union[MNISTBackbone, MLPBackbone],
        num_classes: int = 10,
        num_layers: int = 1,
        eps: float = 1e-6,
        spectral_norm: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.hidden_dim = backbone.out_features
        self.eps = eps
        self.spectral_norm = spectral_norm

        self.class_prototypes = nn.Parameter(
            torch.randn((self.num_classes, self.hidden_dim))
        )

        self.residual_classifier = nn.Sequential(
            *[
                self._spectral_norm(nn.Linear(self.hidden_dim, self.hidden_dim)),
                nn.ReLU(),
            ]
            * num_layers,
            self._spectral_norm(nn.Linear(self.hidden_dim, 1)),
        )
        self.class_probs = nn.Parameter(
            torch.zeros((self.num_classes,)), requires_grad=False
        )
        self.fitted_class_probs = False

    def forward(
        self, x: torch.Tensor, detach_residual: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.get_features(x)
        residuals = self.get_residuals(z)
        logits_y_z = self.classify_residuals(residuals)
        return logits_y_z

    def get_residual_densities(self, x: torch.Tensor) -> torch.Tensor:
        z = self.get_features(x)
        residuals = self.get_residuals(z)
        logits_z_y = self._log_elu(
            self.residual_classifier(residuals)
        )  # (Batch * Class, 1)
        logits_z_y = logits_z_y.reshape(-1, self.num_classes)  # (Batch, Class)
        return logits_z_y

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)  # (Batch, Features)
        return z

    def get_residuals(self, observed_features: torch.Tensor) -> torch.Tensor:
        residuals = (
            observed_features[:, None, :] - self.class_prototypes[None, ...]
        )  # (Batch, Class, Features)
        return residuals

    def classify_residuals(self, residuals: torch.Tensor) -> Tuple[torch.Tensor]:
        residuals = residuals.reshape(-1, self.hidden_dim)  # (Batch * Class, Features)
        logits_z_y = self._log_elu(
            self.residual_classifier(residuals)
        )  # (Batch * Class, 1)
        logits_z_y = logits_z_y.reshape(-1, self.num_classes)  # (Batch, Class)
        logits_joint = self.calculate_joint(logits_z_y)
        logits_z = torch.log(torch.exp(logits_joint).sum(dim=1, keepdims=True))
        logits_y_z = logits_joint - logits_z
        return logits_y_z

    @staticmethod
    def _log_elu(features: torch.Tensor) -> torch.Tensor:
        log_features = torch.log(features + 1)
        return torch.where(features > 0, log_features, features)

    def calculate_joint(self, logits_z_y: torch.Tensor) -> torch.Tensor:
        if not self.fitted_class_probs:
            raise ValueError(
                "Marginal class probabilities have not been estimated.\n"
                "Call 'self.fit_class_probs(TrainDataloader)' first."
            )

        probs_y = torch.clamp(
            self.class_probs, min=self.eps, max=1 - self.eps
        )  # (, Class)
        logits_y = torch.log(probs_y[None, ...])  # (1, Class)
        logits_y = torch.broadcast_to(logits_y, logits_z_y.shape)  # (Batch, Class)
        logits_joint = logits_z_y + logits_y

        return logits_joint

    def fit_class_probs(self, dataloader: DataLoader):
        total = 0
        class_probs = 0
        for x, y in dataloader:
            y = self.move_tensor_to_device(y)

            total += x.shape[0]
            class_probs += torch.nn.functional.one_hot(
                y, num_classes=self.num_classes
            ).sum(dim=0)

        class_probs = class_probs / total
        self.class_probs.copy_(class_probs)

        self.fitted_class_probs = True

        print(f"Class probabilities: {self.class_probs}")

    def move_tensor_to_device(self, x: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        return x.to(device)

    def _spectral_norm(self, layer: nn.Module) -> nn.Module:
        if self.spectral_norm:
            layer = spectral_norm(layer)
        return layer
