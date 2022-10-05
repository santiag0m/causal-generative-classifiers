import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy

from .running_average import RunningAverage


def train(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    optim: torch.optim.Optimizer,
    use_pbar: bool = False,
) -> float:
    model.train()
    device = next(model.parameters()).device

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

        logits = model(inputs)

        ce_loss = cross_entropy(logits, targets)

        ce_loss.backward()
        optim.step()
        ce_loss = ce_loss.item()

        preds = torch.argmax(logits, dim=-1)
        correct = preds == targets
        accuracy.update(correct.cpu())
        avg_acc = accuracy.value

        cum_ce_loss += ce_loss
        avg_ce_loss = cum_ce_loss / (idx + 1)

        if use_pbar:
            pbar.set_description(f"{avg_ce_loss=:.3f} {avg_acc=:.3f}")
    return avg_ce_loss, avg_acc.item()


def eval(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    use_pbar: bool = False,
) -> float:
    model.eval()
    device = next(model.parameters()).device

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

            logits = model(inputs)

            ce_loss = cross_entropy(logits, targets).item()

            preds = torch.argmax(logits, dim=-1)
            correct = preds == targets
            accuracy.update(correct.cpu())
            avg_acc = accuracy.value

            cum_ce_loss += ce_loss
            avg_ce_loss = cum_ce_loss / (idx + 1)

            if use_pbar:
                pbar.set_description(f"[VAL] {avg_ce_loss=:.3f} {avg_acc=:.3f}")
    return avg_ce_loss, avg_acc.item()
