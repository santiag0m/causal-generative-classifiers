import os
import json
import shutil
from typing import List, Dict

import torch
from torchvision import transforms
import pandas as pd
from torch.utils.data import DataLoader, random_split

from lib.utils.hsic_trainer import train, eval
from lib.datasets import ImbalancedImageFolder
from lib.utils.accuracy import compute_accuracy
from lib.models import get_backbone_for_dataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 10

TRIAL_FOLDER = "hsic_trial_results"


def generate_class_unbalance():
    ratios = torch.tensor([1] + [100] * 9, dtype=torch.float32)
    weights = ratios / torch.sum(ratios)
    return {str(i): weights[i] for i in range(NUM_CLASSES)}


def generate_single_class():
    ratios = torch.tensor([1] + [0] * 9, dtype=torch.float32)
    weights = ratios / torch.sum(ratios)
    return {str(i): weights[i] for i in range(NUM_CLASSES)}


def mse_one_hot(
    inputs: torch.tensor, preds: torch.tensor, targets: torch.tensor
) -> torch.Tensor:
    targets = torch.nn.functional.one_hot(targets, num_classes=NUM_CLASSES).float()
    targets = targets.to(preds.device)
    residuals = (targets - preds) ** 2
    residuals = residuals.sum(dim=-1)
    return residuals.mean()


def experiment(
    hidden_dim: int = 10,
    cifar10: bool = False,
    seed: int = 0,
    verbose: bool = True,
    **kwargs,
):
    torch.manual_seed(seed)

    if cifar10:
        epochs = 50
        learning_rate = 0.01
        batch_size = 128
        weight_decay = 0.0001
        momentum = 0.9
        model_name = "cifar10"
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )
    else:
        epochs = 20
        learning_rate = 5e-2
        batch_size = 32
        weight_decay = 0
        momentum = 0
        model_name = "mnist"
        transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
    model = get_backbone_for_dataset(dataset=model_name, hidden_dim=hidden_dim)

    if cifar10:
        model = torch.nn.Sequential(model, torch.nn.Linear(model.out_features, 10))

    model.to(DEVICE)

    # Create Datasets
    source_dataset = ImbalancedImageFolder(
        f"data/class_{model_name}/train",
        class_weights=generate_class_unbalance(),
        seed=seed,
        transform=transform,
    )
    train_dataset, val_dataset, _ = random_split(
        source_dataset, [10_000, 1_000, len(source_dataset) - 11_000]
    )
    target_dataset = ImbalancedImageFolder(
        f"data/class_{model_name}/test",
        class_weights=generate_single_class(),
        seed=seed,
        transform=transform,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    target_dataloader = DataLoader(target_dataset, batch_size=batch_size)

    # Setup Optimizer
    optim = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    # Train
    train_history = []
    val_history = []
    best_loss = 1e10
    for epoch_idx in range(epochs):
        if verbose:
            print(f"\nEpoch {epoch_idx}")
        train_loss = train(
            model=model,
            criterion=mse_one_hot,
            dataloader=train_dataloader,
            optim=optim,
            use_pbar=verbose,
        )
        val_loss = eval(
            model=model,
            criterion=mse_one_hot,
            dataloader=val_dataloader,
            use_pbar=verbose,
        )
        train_history.append(train_loss)
        val_history.append(val_loss)

        if val_loss <= best_loss:
            torch.save(model.state_dict(), "./best.pth")
            best_loss = val_loss

    # Check accuracy
    model.predict = model.forward
    train_accuracy = compute_accuracy(model, train_dataloader)
    val_accuracy = compute_accuracy(model, val_dataloader)
    target_accuracy = compute_accuracy(model, target_dataloader)

    print(f"{train_accuracy=:.4f}, {val_accuracy=:.4f}, {target_accuracy=:.4f}\n")

    results = {
        "train_history": train_history,
        "val_history": val_history,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "target_accuracy": target_accuracy,
    }

    return results


def multiple_trials(experiment_config: Dict, num_trials: int) -> Dict:
    if os.path.isdir(TRIAL_FOLDER):
        shutil.rmtree(TRIAL_FOLDER)
    os.makedirs(TRIAL_FOLDER)

    results = []
    for i in range(num_trials):
        print(f"Experiment {i+1}/{num_trials}")
        trial_results = experiment(**experiment_config, seed=i)
        results.append(trial_results)
        with open(os.path.join(TRIAL_FOLDER, f"trial_{i:03d}.json"), "w") as f:
            json.dump(trial_results, f)

    train_accuracy = [trial["train_accuracy"] for trial in results]
    val_accuracy = [trial["val_accuracy"] for trial in results]
    target_accuracy = [trial["target_accuracy"] for trial in results]

    results = {
        "train": pd.Series(train_accuracy)
        .fillna(0)
        .rename(experiment_config["model_name"]),
        "val": pd.Series(val_accuracy)
        .fillna(0)
        .rename(experiment_config["model_name"]),
        "target": pd.Series(target_accuracy)
        .fillna(0)
        .rename(experiment_config["model_name"]),
    }

    return results


def group_results(results: List[Dict]) -> pd.DataFrame:
    keys = results[0].keys()

    df_list = []
    for key in keys:
        df = pd.concat([exp_res[key] for exp_res in results], axis=1)
        df = (
            df.stack()
            .rename("Accuracy")
            .rename_axis(index=["exp", "model_name"])
            .reset_index()
        )
        df["setting"] = key
        df_list.append(df)
    df = pd.concat(df_list)

    return df


def main(
    cifar10: bool = False,
    hidden_dim: int = 10,
    num_trials: int = 20,
):
    models = [
        {"model_name": "CNN", "cnn": True},
    ]

    if cifar10:
        suffix = "cifar10"
    else:
        suffix = "mnist"

    results = []
    for model_config in models:
        experiment_config = {"cifar10": cifar10, "hidden_dim": hidden_dim}
        experiment_config = {**experiment_config, **model_config}
        exp_results = multiple_trials(
            num_trials=num_trials, experiment_config=experiment_config
        )
        results.append(exp_results)
    results = group_results(results)
    results.to_csv(f"l2_results_{suffix}.csv", index=False)


if __name__ == "__main__":
    main()
