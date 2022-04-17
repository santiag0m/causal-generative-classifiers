from typing import List, Union

from .cnn import CNNBackbone
from .mlp import MLPBackbone
from .generative_features import GenerativeFeatures


def get_backbone(
    cnn: bool = False,
    mlp_layers: List[int] = [],
) -> Union[MLPBackbone, CNNBackbone]:
    # Init model
    if cnn:
        if mlp_layers:
            raise ValueError(
                "Conflicting models. Either set `cnn` to false or `mlp_layers` to `[]`"
            )
        model = CNNBackbone()
    elif mlp_layers:
        model = MLPBackbone(in_features=28 * 28, layers=mlp_layers)
    else:
        raise ValueError(
            "No model parameters were provided, please provide values "
            "for either `cnn` OR `mlp_layers` parameters"
        )
    return model
