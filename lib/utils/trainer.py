from typing import Callable, Optional

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy

from .mmdm import MMDMOptim
from .running_average import RunningAverage
from lib.losses.hsic import hsic_residuals


def train(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    mmdm_optim: MMDMOptim,
    use_pbar: bool = False,
    use_hsic: bool = True,
) -> float:
    model.train()
    device = next(model.parameters()).device

    cum_hsic_loss = 0
    cum_ce_loss = 0
    if use_pbar:
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    else:
        pbar = enumerate(dataloader)

    accuracy = RunningAverage()
    for idx, (inputs, targets) in pbar:
        mmdm_optim.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)

        features = model.get_features(inputs)
        residuals = model.get_residuals(features)
        logits = model.classify_residuals(residuals)

        ce_loss = cross_entropy(logits, targets)

        if use_hsic:
            hsic_loss = hsic_residuals(residuals, targets, featurewise=False)
        else:
            hsic_loss = -1

        if use_hsic:
            loss = mmdm_optim.lagrangian(
                main_loss=ce_loss, constrained_loss=hsic_loss, target_value=0
            )
            loss.backward()
            mmdm_optim.step()

            hsic_loss = hsic_loss.item()
            ce_loss = ce_loss.item()
        else:
            ce_loss.backward()
            mmdm_optim.model_optim.step()
            ce_loss = ce_loss.item()

        preds = torch.argmax(logits, dim=-1)
        correct = preds == targets
        accuracy.update(correct.cpu())
        avg_acc = accuracy.value

        cum_hsic_loss += hsic_loss
        cum_ce_loss += ce_loss

        avg_hsic_loss = cum_hsic_loss / (idx + 1)
        avg_ce_loss = cum_ce_loss / (idx + 1)

        if use_pbar:
            pbar.set_description(
                f"{avg_hsic_loss=:.3f} {avg_ce_loss=:.3f} {avg_acc=:.3f}"
            )
    return avg_hsic_loss, avg_ce_loss, avg_acc.item()


def eval(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    use_pbar: bool = False,
    use_hsic: bool = True,
) -> float:
    model.eval()
    device = next(model.parameters()).device

    cum_hsic_loss = 0
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

            features = model.get_features(inputs)
            residuals = model.get_residuals(features)
            logits = model.classify_residuals(residuals)

            if use_hsic:
                hsic_loss = hsic_residuals(residuals, targets).item()
            else:
                hsic_loss = -1

            ce_loss = cross_entropy(logits, targets).item()

            preds = torch.argmax(logits, dim=-1)
            correct = preds == targets
            accuracy.update(correct.cpu())
            avg_acc = accuracy.value

            cum_hsic_loss += hsic_loss
            cum_ce_loss += ce_loss

            avg_hsic_loss = cum_hsic_loss / (idx + 1)
            avg_ce_loss = cum_ce_loss / (idx + 1)

            if use_pbar:
                pbar.set_description(
                    f"[VAL] {avg_hsic_loss=:.3f} {avg_ce_loss=:.3f} {avg_acc=:.3f}"
                )
    return avg_hsic_loss, avg_ce_loss, avg_acc.item()
