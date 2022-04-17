from typing import Union

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.neighbors import KernelDensity

from .cnn import CNNBackbone
from .mlp import MLPBackbone


class GenerativeFeatures(nn.Module):
    def __init__(
        self, backbone: Union[CNNBackbone, MLPBackbone], num_classes: int = 10
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        self.embedding_layer = nn.Embedding(
            num_embeddings=num_classes, embedding_dim=backbone.out_features
        )
        # Every feature for every class has a KDE
        self.kde_list = [
            [None for _ in range(backbone.out_features)]
            for _ in range(self.num_classes)
        ]
        self.class_probs = nn.Parameter(
            torch.zeros((self.num_classes,)), requires_grad=False
        )

        self.fitted = False

    def normalize_embeddings(self):
        with torch.no_grad():
            # Normalize class embeddings to avoid empty features
            embeddings = self.embedding_layer.weight
            norm = torch.norm(embeddings, dim=-1, keepdim=True)
            embeddings = embeddings / norm
            self.embedding_layer.weight.copy_(embeddings)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.normalize_embeddings()
        observed_features = self.backbone(x)
        predicted_features = self.embedding_layer(y)
        residual = observed_features - predicted_features
        return residual

    def fit_kde(self, dataloader: DataLoader, kernel_bandwidth: float = 1.0):
        residual_list = [[] for _ in range(self.num_classes)]
        with torch.no_grad():
            total = 0
            class_probs = torch.zeros_like(self.class_probs)
            for x, y in tqdm(dataloader, total=len(dataloader)):
                x = self.move_tensor_to_device(x)
                y = self.move_tensor_to_device(y)
                residual = self.forward(x, y)
                for i in range(self.num_classes):
                    class_idx = y == i
                    class_residual = residual[class_idx]
                    residual_list[i].append(class_residual)
                total += x.shape[0]
                class_probs += torch.nn.functional.one_hot(
                    y, num_classes=self.num_classes
                ).sum(dim=0)
        class_probs = class_probs / total
        self.class_probs.copy_(class_probs)
        residual_list = [
            torch.cat(class_residuals).cpu().numpy()
            for class_residuals in residual_list
        ]
        for i in range(self.num_classes):
            class_residual = residual_list[i]
            for j in range(self.backbone.out_features):
                self.kde_list[i][j] = KernelDensity(
                    kernel="gaussian", bandwidth=kernel_bandwidth
                ).fit(class_residual[:, [j]])
        self.fitted = True

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        class_probs = []
        # Calculate joint probability for each class
        for i in range(self.num_classes):
            with torch.no_grad():
                residuals = self.forward(x, self.move_tensor_to_device(torch.tensor(i)))
                class_probs.append(self.joint_class_probability(residuals, class_idx=i))
        # Marginalize over Y
        class_probs = torch.stack(class_probs, dim=1)
        marginal = class_probs.sum(axis=1, keepdim=True)
        return class_probs / marginal

    def joint_class_probability(
        self,
        residuals: torch.Tensor,
        class_idx: int,
    ) -> torch.Tensor:
        # Get class probability
        log_likelihood = torch.ones(
            (residuals.shape[0],), device=residuals.device
        ) * torch.log(self.class_probs[class_idx])

        # Calculate class conditional probability of each feature
        feature_kdes = self.kde_list[class_idx]
        if residuals.shape[1] != len(feature_kdes):
            raise ValueError(
                "Residuals are not consistent with the features"
                f"Expected {len(feature_kdes)} but got {residuals.shape[1]}."
            )
        for i, kde_estimator in enumerate(feature_kdes):
            feat_residuals = residuals[:, [i]].cpu().numpy()
            feat_log_likelihood = kde_estimator.score_samples(feat_residuals)
            log_likelihood += self.move_tensor_to_device(
                torch.tensor(feat_log_likelihood)
            )

        return torch.exp(log_likelihood)

    def move_tensor_to_device(self, x: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        return x.to(device)
