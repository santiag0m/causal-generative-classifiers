from typing import List

import torch
from torch.utils.data import DataLoader, random_split

from lib.losses import HSIC
from lib.datasets import MNIST
from lib.utils.trainer import train, eval
from lib.utils.accuracy import compute_accuracy
from lib.models import get_backbone, GenerativeFeatures


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 10


def hsic_one_hot(residuals: torch.tensor, targets: torch.tensor) -> torch.Tensor:
    targets = torch.nn.functional.one_hot(targets, num_classes=NUM_CLASSES).float()
    targets = targets.to(residuals.device)
    return HSIC(residuals, targets)


def experiment(
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    cnn: bool = False,
    mlp_layers: List[int] = [],
    verbose: bool = True,
):
    backbone = get_backbone(cnn=cnn, mlp_layers=mlp_layers)
    model = GenerativeFeatures(backbone, NUM_CLASSES)
    model.to(DEVICE)

    # Create Datasets
    source_dataset = MNIST(rotated=False, train=True)
    train_dataset, val_dataset, _ = random_split(
        source_dataset, [10_000, 1_000, len(source_dataset) - 11_000]
    )
    target_dataset = MNIST(rotated=True, train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    target_dataloader = DataLoader(target_dataset, batch_size=batch_size)

    # Setup Optimizer
    optim = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    # Train
    train_history = []
    val_history = []
    best_loss = 1e10
    for epoch_idx in range(num_epochs):
        if verbose:
            print(f"Epoch {epoch_idx}")
        train_loss = train(
            model=model,
            criterion=hsic_one_hot,
            dataloader=train_dataloader,
            optim=optim,
            use_pbar=verbose,
        )
        val_loss = eval(
            model=model,
            criterion=hsic_one_hot,
            dataloader=val_dataloader,
            use_pbar=verbose,
        )
        train_history.append(train_loss)
        val_history.append(val_loss)

        if val_loss <= best_loss:
            torch.save(model.state_dict(), "./best.pth")
            best_loss = val_loss

    # Fit KDE after training
    model.fit_kde(train_dataloader)

    # Check accuracy
    train_accuracy = compute_accuracy(model, train_dataloader)
    val_accuracy = compute_accuracy(model, val_dataloader)
    target_accuracy = compute_accuracy(model, target_dataloader)

    print(f"{train_accuracy=}, {val_accuracy=}, {target_accuracy=}")


def main(batch_size: int = 32, learning_rate: float = 1e-5, num_epochs: int = 100):
    experiment(
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        cnn=True,
    )


if __name__ == "__main__":
    main()
