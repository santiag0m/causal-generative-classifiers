from typing import List, Union

from .cifar10_cnn import CIFAR10Backbone, ResNet
from .mnist_cnn import MNISTBackbone
from .cgc_residual import CGCResidual
from .cgc_kde import CGCKDE
from .dot_classifier import DotClassifier


def get_backbone(
    model_name: str = "mnist", hidden_dim: int = 10
) -> Union[MNISTBackbone, CIFAR10Backbone]:
    # Init model
    if model_name == "mnist":
        model = MNISTBackbone(out_features=hidden_dim)
    elif model_name == "cifar10":
        model = ResNet(n=3)  # CIFAR10Backbone(out_features=hidden_dim)
    else:
        raise ValueError(
            "No model parameters were provided, please provide values "
            "for either `cnn` OR `mlp_layers` parameters"
        )
    return model
