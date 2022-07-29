from typing import Union, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.neighbors import KernelDensity

from .cnn import CNNBackbone
from .mlp import MLPBackbone
from .utils import multiply_probs_with_logits, divide_probs_with_logits


class GenerativeFeatures(nn.Module):
    def __init__(
        self, backbone: Union[CNNBackbone, MLPBackbone], num_classes: int = 10, eps: float = 1e-6
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.hidden_dim = backbone.out_features
        self.eps = eps

        self.class_prototypes = nn.Parameter(
            torch.randn((self.num_classes, self.hidden_dim))
        )
        self.residual_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        self.class_probs = nn.Parameter(
            torch.zeros((self.num_classes,)), requires_grad=False
        )
        self.fitted_class_probs = False

    def forward(self, x: torch.Tensor, detach_residual: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        residuals = self.get_residuals(x)
        logits_y_x = self.classify_residuals(residuals, detach_residual)
        return residuals, logits_y_x

    def get_residuals(self, x: torch.Tensor) -> torch.Tensor:
        observed_features = self.backbone(x)  # (Batch, Features)
        residuals = observed_features[:, None, :] - self.class_prototypes[None, ...]  # (Batch, Class, Features)
        return residuals

    def classify_residuals(self, residuals: torch.Tensor, detach_residual: bool = True) -> Tuple[torch.Tensor]:
        if detach_residual:
            residuals = residuals.detach()
        residuals = residuals.reshape(-1, self.hidden_dim)  # (Batch * Class, Features)
        logits_x_y = self.residual_classifier(residuals)  # (Batch * Class, 1)
        logits_x_y = logits_x_y.reshape(-1, self.num_classes)  # (Batch, Class)
        logits_y_x = self.calculate_posterior(logits_x_y)
        return logits_y_x


    def calculate_posterior(self, logits_x_y: torch.Tensor) -> torch.Tensor:
        if not self.fitted_class_probs:
            raise ValueError(
                "Marginal class probabilities have not been estimated.\n"
                "Call 'self.fit_class_probs(TrainDataloader)' first."
                )
        probs_y = torch.clamp(self.class_probs, min=self.eps, max=1-self.eps)  # (, Class)
        logits_y = torch.logit(probs_y[None, ...])  # (1, Class)
        logits_y = torch.broadcast_to(logits_y, logits_x_y.shape)  # (Batch, Class)
        logits_joint = multiply_probs_with_logits(logits_x_y, logits_y)  # (Batch, Class)
        probs_joint = torch.sigmoid(logits_joint)
        probs_x = torch.sum(probs_joint, dim=-1, keepdim=True)  # (Batch, 1)
        logits_x = torch.logit(probs_x)
        logits_x = torch.broadcast_to(logits_x, logits_joint.shape)  # (Batch, Class)
        logits_y_x = divide_probs_with_logits(logits_joint, logits_x)
        return logits_y_x

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
