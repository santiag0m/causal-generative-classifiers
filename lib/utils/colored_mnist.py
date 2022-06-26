# Modified from: https://github.com/facebookresearch/InvariantRiskMinimization/blob/main/code/colored_mnist/main.py
from typing import List, Tuple, Union, Callable

import torch


def make_environment(
    images: torch.Tensor, labels: torch.Tensor, e: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    def torch_bernoulli(p, size):
        return (torch.rand(size) < p).float()

    def torch_xor(a, b):
        return (a - b).abs()  # Assumes both inputs are either 0 or 1

    # Assign a binary label based on the digit; flip label with probability 0.25
    labels = (labels < 5).float()
    labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
    # Assign a color based on the label; flip the color with probability e
    colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
    # Apply the color to the image by zeroing out the other color channel
    images = torch.stack([images, images], dim=1)
    images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0
    return images, labels.long()


def make_collate_fn(e: Union[float, List[float]]) -> Callable:
    if isinstance(e, float):
        e = [e]

    idx = IntWrapper(value=0)

    def collate_fn(
        samples: List[Tuple[torch.Tensor, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        images, labels = zip(*samples)
        images = torch.cat(images, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)
        images, labels = make_environment(images=images, labels=labels, e=e[idx.value])

        idx.update((idx.value + 1) % len(e))

        return images, labels

    return collate_fn


class IntWrapper:
    def __init__(self, value: int):
        self.value = value

    def update(self, value: int):
        self.value = value
