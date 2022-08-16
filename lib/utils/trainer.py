from typing import Callable, Optional

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy

from .mmdm import MMDMOptim
from .running_average import RunningAverage


def shuffle_batch(x: torch.Tensor) -> torch.Tensor:
    batch = x.shape[0]
    indices = torch.randperm(batch)
    return x[indices, ...]


def nll_diff(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    nll_marginals = cross_entropy(
        logits,
        shuffle_batch(targets),
    )
    nll_joint = cross_entropy(logits, targets)
    return nll_marginals - nll_joint


def train(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    adversarial_optim: torch.optim.Optimizer,
    mmdm_optim: MMDMOptim,
    use_pbar: bool = False,
    use_adversarial: bool = True,
) -> float:
    model.train()
    device = next(model.parameters()).device

    cum_adversarial_loss = 0
    cum_ce_loss = 0
    if use_pbar:
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    else:
        pbar = enumerate(dataloader)

    accuracy = RunningAverage()
    for idx, (inputs, targets) in pbar:
        adversarial_optim.zero_grad()
        mmdm_optim.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)

        features = model.get_features(inputs)
        residuals = model.get_residuals(features)
        logits = model.classify_residuals(residuals)

        ce_loss = cross_entropy(logits, targets)

        if use_adversarial:
            prototypes = model.class_prototypes(
                torch.nn.functional.one_hot(
                    targets, num_classes=model.num_classes
                ).float()
            )
            adversarial_residual = features - prototypes

            if idx % 2:
                adversarial_loss = nll_diff(
                    model.adversarial_classifier(adversarial_residual.detach()), targets
                )
                adversarial_loss.backward()
                adversarial_optim.step()
            else:
                adversarial_loss = nll_diff(
                    model.adversarial_classifier(adversarial_residual), targets
                )
                mmdm_loss = mmdm_optim.lagrangian(
                    main_loss=ce_loss,
                    constrained_loss=-1 * adversarial_loss,
                    target_value=0,
                )
                mmdm_loss.backward()
                mmdm_optim.step()

            adversarial_loss = adversarial_loss.item()
        else:
            ce_loss.backward()
            mmdm_optim.model_optim.step()

        ce_loss = ce_loss.item()

        preds = torch.argmax(logits, dim=-1)
        correct = preds == targets
        accuracy.update(correct.cpu())
        avg_acc = accuracy.value

        cum_adversarial_loss += adversarial_loss
        cum_ce_loss += ce_loss

        avg_adversarial_loss = cum_adversarial_loss / (idx + 1)
        avg_ce_loss = cum_ce_loss / (idx + 1)

        if use_pbar:
            pbar.set_description(
                f"{avg_adversarial_loss=:.3f} {avg_ce_loss=:.3f} {avg_acc=:.3f}"
            )
    return avg_adversarial_loss, avg_ce_loss, avg_acc.item()


def eval(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    use_pbar: bool = False,
    use_adversarial: bool = True,
) -> float:
    model.eval()
    device = next(model.parameters()).device

    cum_adversarial_loss = 0
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

            if use_adversarial:
                prototypes = model.class_prototypes(
                    torch.nn.functional.one_hot(
                        targets, num_classes=model.num_classes
                    ).float()
                )
                adversarial_residual = features - prototypes
                adversarial_loss = nll_diff(
                    model.adversarial_classifier(adversarial_residual).detach(), targets
                ).item()
            else:
                adversarial_loss = -1

            ce_loss = cross_entropy(logits, targets).item()

            preds = torch.argmax(logits, dim=-1)
            correct = preds == targets
            accuracy.update(correct.cpu())
            avg_acc = accuracy.value

            cum_adversarial_loss += adversarial_loss
            cum_ce_loss += ce_loss

            avg_adversarial_loss = cum_adversarial_loss / (idx + 1)
            avg_ce_loss = cum_ce_loss / (idx + 1)

            if use_pbar:
                pbar.set_description(
                    f"[VAL] {avg_adversarial_loss=:.3f} {avg_ce_loss=:.3f} {avg_acc=:.3f}"
                )
    return avg_adversarial_loss, avg_ce_loss, avg_acc.item()
