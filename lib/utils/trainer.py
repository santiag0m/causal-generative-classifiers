from typing import Callable, Optional

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy

from lib.losses.hsic import HSIC, hsic_one_hot
from .running_average import RunningAverage

def train(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    optim: torch.optim.Optimizer,
    use_pbar: bool = False,
    num_classes: int = 10,
) -> float:
    model.train()
    device = next(model.parameters()).device
    cum_loss = 0
    if use_pbar:
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    else:
        pbar = enumerate(dataloader)

    accuracy = RunningAverage()
    for idx, (inputs, targets) in pbar:
        optim.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)
        residuals, logits = model(inputs)
        loss = hsic_one_hot(residuals, targets)
        loss += cross_entropy(logits, targets)

        mapped_feats = model.class_prototypes[targets, :]
        label_loss = -1 * HSIC(
                mapped_feats,
                torch.nn.functional.one_hot(
                    targets, num_classes=num_classes
                ).float(),
            )

        loss = loss + label_loss
        loss.backward()
        optim.step()

        preds = torch.argmax(logits, dim=-1)
        correct = preds == targets
        accuracy.update(correct.cpu())

        cum_loss += loss.item()
        avg_loss = cum_loss / (idx + 1)
        if use_pbar:
            pbar.set_description(f"Loss: {avg_loss:.4f} - Acc:{accuracy.value:.4f}")
    return avg_loss, accuracy.value


def eval(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    use_pbar: bool = False,
    num_classes: int = 10,
) -> float:
    model.eval()
    device = next(model.parameters()).device
    cum_loss = 0
    if use_pbar:
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    else:
        pbar = enumerate(dataloader)

    with torch.no_grad():
        accuracy = RunningAverage()
        for idx, (inputs, targets) in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            residuals, logits = model(inputs)
            loss = hsic_one_hot(residuals, targets)
            loss += cross_entropy(logits, targets)

            mapped_feats = model.class_prototypes[targets, :]
            label_loss = -1 * HSIC(
                mapped_feats,
                torch.nn.functional.one_hot(
                    targets, num_classes=num_classes
                ).float(),
            )

            loss = loss + label_loss

            preds = torch.argmax(logits, dim=-1)
            correct = preds == targets
            accuracy.update(correct.cpu())

            cum_loss += loss.item()
            avg_loss = cum_loss / (idx + 1)
            if use_pbar:
                pbar.set_description(f"Loss: {avg_loss:.4f} - Acc:{accuracy.value:.4f}")
    return avg_loss, accuracy.value
