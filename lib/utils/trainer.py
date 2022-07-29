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
    train_backbone: bool = True,
    train_classifier: bool = True,
    detach_residual: bool = True,
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

        loss = 0

        if train_backbone:
            residuals = model.get_residuals(inputs)
            loss += hsic_one_hot(residuals, targets)
            mapped_feats = model.class_prototypes[targets, :]
            loss -= HSIC(
                    mapped_feats,
                    torch.nn.functional.one_hot(
                        targets, num_classes=num_classes
                    ).float(),
                )
        else:
            with torch.no_grad():
                residuals = model.get_residuals(inputs)
        if train_classifier:
            logits = model.clasify_residuals(residuals, detach_residual=detach_residual)
            loss += cross_entropy(logits, targets)
        
        loss.backward()
        optim.step()

        if train_classifier:
            preds = torch.argmax(logits, dim=-1)
            correct = preds == targets
            accuracy.update(correct.cpu())
            avg_acc = accuracy.value
        else:
            avg_acc = -1

        cum_loss += loss.item()
        avg_loss = cum_loss / (idx + 1)
        if use_pbar:
            pbar.set_description(f"Loss: {avg_loss:.4f} - Acc:{avg_acc:.4f}")
    return avg_loss, avg_acc


def eval(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    use_pbar: bool = False,
    num_classes: int = 10,
    eval_backbone: bool = True,
    eval_classifier: bool = True
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

            loss = 0
            with torch.no_grad():
                residuals = model.get_residuals(inputs)
            if eval_backbone:
                loss += hsic_one_hot(residuals, targets)
                mapped_feats = model.class_prototypes[targets, :]
                loss -= HSIC(
                        mapped_feats,
                        torch.nn.functional.one_hot(
                            targets, num_classes=num_classes
                        ).float(),
                    )

            with torch.no_grad():
                logits = model.clasify_residuals(residuals)
            if eval_classifier:
                loss += cross_entropy(logits, targets)

            preds = torch.argmax(logits, dim=-1)
            correct = preds == targets
            accuracy.update(correct.cpu())

            cum_loss += loss.item()
            avg_loss = cum_loss / (idx + 1)
            if use_pbar:
                pbar.set_description(f"Loss: {avg_loss:.4f} - Acc:{accuracy.value:.4f}")
    return avg_loss, accuracy.value
