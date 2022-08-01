from typing import Callable, Optional

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy

from .running_average import RunningAverage
from lib.losses.hsic import HSIC, hsic_residuals, hsic_prototypes, hsic_independence


def train(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    optim: torch.optim.Optimizer,
    use_pbar: bool = False,
    num_classes: int = 10,
    train_backbone: bool = True,
    train_classifier: bool = True,
    only_cross_entropy: bool = False
) -> float:
    model.train()
    device = next(model.parameters()).device
    detach_residual = not only_cross_entropy

    cum_loss = 0
    cum_hsic_loss = 0
    cum_label_loss = 0
    cum_ce_loss = 0
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

        residuals = model.get_residuals(inputs)

        if only_cross_entropy:
            hsic_loss = -1
            label_loss = -1
        else:
            hsic_loss = hsic_residuals(residuals, targets)
            label_loss = hsic_prototypes(model.class_prototypes, targets)
            indep_loss = hsic_independence(residuals, targets)
            loss += hsic_loss + label_loss
            hsic_loss = hsic_loss.item()
            label_loss = label_loss.item()

        if train_classifier:
            logits = model.classify_residuals(residuals, detach_residual=not only_cross_entropy)
            ce_loss = cross_entropy(logits, targets)
            loss += ce_loss
            ce_loss = ce_loss.item()
        else:
            ce_loss = -1
        
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
        cum_hsic_loss += hsic_loss
        cum_label_loss += label_loss
        cum_ce_loss += ce_loss

        avg_loss = cum_loss / (idx + 1)
        avg_hsic_loss = cum_hsic_loss / (idx + 1)
        avg_label_loss = cum_label_loss / (idx + 1)
        avg_ce_loss = cum_ce_loss / (idx + 1)

        if use_pbar:
            pbar.set_description(f"{avg_hsic_loss=:.3f} {avg_label_loss=:.3f} {avg_ce_loss=:.3f} {avg_acc=:.3f}")
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
    cum_hsic_loss = 0
    cum_label_loss = 0
    cum_ce_loss = 0
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
                hsic_loss = hsic_residuals(residuals, targets)
                label_loss = hsic_prototypes(model.class_prototypes, targets)
                indep_loss = hsic_independence(residuals, targets)
                loss += hsic_loss + label_loss
                hsic_loss = hsic_loss.item()
                label_loss = label_loss.item()
            else:
                hsic_loss = -1
                label_loss = -1

            with torch.no_grad():
                logits = model.classify_residuals(residuals)
            if eval_classifier:
                ce_loss = cross_entropy(logits, targets)
                loss += ce_loss
            else:
                ce_loss = -1

            if eval_classifier:
                preds = torch.argmax(logits, dim=-1)
                correct = preds == targets
                accuracy.update(correct.cpu())
                avg_acc = accuracy.value
            else:
                avg_acc = -1

            cum_loss += loss.item()
            cum_hsic_loss += hsic_loss
            cum_label_loss += label_loss
            cum_ce_loss += ce_loss

            avg_loss = cum_loss / (idx + 1)
            avg_hsic_loss = cum_hsic_loss / (idx + 1)
            avg_label_loss = cum_label_loss / (idx + 1)
            avg_ce_loss = cum_ce_loss / (idx + 1)

            if use_pbar:
                pbar.set_description(f"[VAL] {avg_hsic_loss=:.3f} {avg_label_loss=:.3f} {avg_ce_loss=:.3f} {avg_acc=:.3f}")
    return avg_loss, avg_acc
