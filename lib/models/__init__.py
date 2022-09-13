from typing import List, Union

from .cnn import CNNBackbone
from .mlp import MLPBackbone
<<<<<<< HEAD
from .cgc_residual import CGCResidual
=======
from .cgc_kde import CGCKDE
>>>>>>> main


def get_backbone(
    cnn: bool = False,
    in_channels: int = 1,
    in_features: int = 28 * 28,
    mlp_layers: List[int] = [],
    spectral_norm: bool = False,
) -> Union[MLPBackbone, CNNBackbone]:
    # Init model
    if cnn:
        if mlp_layers:
            raise ValueError(
                "Conflicting models. Either set `cnn` to false or `mlp_layers` to `[]`"
            )
        model = CNNBackbone(
            in_channels=in_channels,
            spectral_norm=spectral_norm,
        )
    elif mlp_layers:
        model = MLPBackbone(in_features=in_features, layers=mlp_layers)
    else:
        raise ValueError(
            "No model parameters were provided, please provide values "
            "for either `cnn` OR `mlp_layers` parameters"
        )
    return model
