from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy

from lib.models import CGCResidual
from lib.losses.hsic import hsic_residuals
from .mmdm import MMDMOptim
from .running_average import RunningAverage

def train(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    optim: MMDMOptim,
    use_pbar: bool = False,
    use_hsic: bool = True,
    class_weights: Optional[torch.Tensor] = None,
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
        optim.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)

        if use_hsic:
            if not isinstance(model, CGCResidual):
                raise TypeError(f"Expected model of class 'CGCResidual', got '{type(model)}'")
            
            z = model.get_features(inputs)
            residuals = model.get_residuals(z)
            logits = model.classify_residuals(residuals)

            # MMDM Update
            hsic_loss = hsic_residuals(residuals, targets, featurewise=False)
            ce_loss = cross_entropy(logits, targets, weight=class_weights)
            loss = optim.lagrangian(
                main_loss=ce_loss, constrained_loss=hsic_loss, target_value=0
            )
            loss.backward()
            optim.step()

            hsic_loss = hsic_loss.item()
            cum_hsic_loss += hsic_loss
        else:
            # Normal update
            logits = model(inputs)
            ce_loss = cross_entropy(logits, targets, weight=class_weights)
            ce_loss.backward()
            optim.model_optim.step()
        
        ce_loss = ce_loss.item()

        preds = torch.argmax(logits, dim=-1)
        correct = preds == targets
        accuracy.update(correct.cpu())
        avg_acc = accuracy.value

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
    return_confusion_matrix: bool = False,
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

        confusion_matrix = 0

        for idx, (inputs, targets) in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            if isinstance(model, CGCResidual):
                z = model.get_features(inputs)
                residuals = model.get_residuals(z)
                logits = model.classify_residuals(residuals)

                hsic_loss = hsic_residuals(residuals, targets, featurewise=False).item()
                cum_hsic_loss += hsic_loss
            else:
                logits = model(inputs)

            ce_loss = cross_entropy(logits, targets).item()

            preds = torch.argmax(logits, dim=-1)
            correct = preds == targets
            accuracy.update(correct.cpu())
            avg_acc = accuracy.value

            if return_confusion_matrix:
                batch_confusion_matrix = torch.sparse_coo_tensor(
                    indices=torch.stack([preds, targets], dim=1).T,
                    values=torch.ones(inputs.shape[0], device=preds.device),
                    size=(logits.shape[-1], logits.shape[-1]),
                )
                batch_confusion_matrix = batch_confusion_matrix.coalesce().to_dense()
                confusion_matrix += batch_confusion_matrix

            cum_ce_loss += ce_loss
            avg_hsic_loss = cum_hsic_loss / (idx + 1)
            avg_ce_loss = cum_ce_loss / (idx + 1)

            if use_pbar:
                pbar.set_description(
                    f"[VAL] {avg_hsic_loss=:.3f} {avg_ce_loss=:.3f} {avg_acc=:.3f}"
                )

    if return_confusion_matrix:
        return avg_hsic_loss, avg_ce_loss, avg_acc.item(), confusion_matrix
    else:
        return avg_hsic_loss, avg_ce_loss, avg_acc.item()
