import os
import json
import shutil
from typing import List, Dict

import torch
import pandas as pd
from torch.optim import SGD
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from lib.utils.ce_trainer import train, eval
from lib.datasets import ImbalancedImageFolder
from lib.models import get_backbone, CGCResidual, DotClassifier
from lib.utils.expectation_maximization import expectation_maximization

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 10

TRIAL_FOLDER = "trial_results"


def generate_uniform_weights():
    return {str(i): 1.0 / NUM_CLASSES for i in range(NUM_CLASSES)}


def generate_random_weights():
    weights = torch.softmax(torch.randn(NUM_CLASSES), 0)
    return {str(i): weights[i] for i in range(NUM_CLASSES)}


def generate_class_unbalance():
    ratios = torch.tensor([1] + [100] * 9, dtype=torch.float32)
    weights = ratios / torch.sum(ratios)
    return {str(i): weights[i] for i in range(NUM_CLASSES)}


def generate_single_class():
    ratios = torch.tensor([1] + [0] * 9, dtype=torch.float32)
    weights = ratios / torch.sum(ratios)
    return {str(i): weights[i] for i in range(NUM_CLASSES)}


def experiment(
    hidden_dim: int = 10,
    cifar10: bool = False,
    spectral_norm: bool = False,
    verbose: bool = True,
    seed: int = 0,
    use_residual: bool = False,
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
    backbone = get_backbone(model_name=model_name, hidden_dim=hidden_dim)

    if use_residual:
        print("Using residual")
        model = CGCResidual(
            backbone, NUM_CLASSES, spectral_norm=spectral_norm, num_layers=1
        )
    else:
        model = DotClassifier(backbone, NUM_CLASSES, num_layers=1)
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

    if use_residual:
        # Fit class priors before training
        model.fit_class_probs(train_dataloader)

    # Train
    train_history = {"cross_entropy": []}
    val_history = {"cross_entropy": []}
    best_loss = 1e10

    for epoch_idx in range(epochs):
        if verbose:
            print(f"\nEpoch {epoch_idx}")
        train_ce_loss, train_accuracy = train(
            model=model,
            dataloader=train_dataloader,
            optim=SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            ),
            use_pbar=verbose,
        )
        val_ce_loss, val_accuracy = eval(
            model=model,
            dataloader=val_dataloader,
            use_pbar=verbose,
        )
        train_history["cross_entropy"].append(train_ce_loss)
        val_history["cross_entropy"].append(val_ce_loss)

        if val_accuracy <= best_loss:
            torch.save(model.state_dict(), "./best.pth")
            best_loss = val_accuracy

    # Adjust marginal
    if use_residual:
        _, y_marginal = expectation_maximization(model, target_dataloader)
        print(y_marginal)
        model.class_probs.copy_(y_marginal)

    # Check accuracy
    target_ce_loss, target_accuracy = eval(
        model=model,
        dataloader=target_dataloader,
        use_pbar=verbose,
    )

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
    use_residual: bool = False,
    spectral_norm: bool = False,
):
    models = [
        {"model_name": "CNN", "cnn": True},
    ]

    results = []
    for model_config in models:
        experiment_config = {
            "cifar10": cifar10,
            "use_residual": use_residual,
            "spectral_norm": spectral_norm,
            "hidden_dim": hidden_dim,
        }
        experiment_config = {**experiment_config, **model_config}
        exp_results = multiple_trials(
            num_trials=num_trials, experiment_config=experiment_config
        )
        results.append(exp_results)
    results = group_results(results)
    if use_residual:
        results.to_csv("ce_results_residual.csv", index=False)
    else:
        results.to_csv("ce_results.csv", index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment setup")
    parser.add_argument("--use_residual", action="store_true")
    parser.add_argument("--spectral_norm", action="store_true")
    args = parser.parse_args()
    main(use_residual=args.use_residual, spectral_norm=args.spectral_norm)
