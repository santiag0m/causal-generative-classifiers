import os
import json
import shutil
from typing import List, Dict

import torch
import pandas as pd
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split

from lib.utils import label_shift
from lib.hyperparams import HYPERPARAMS
from lib.utils.ce_trainer import train, eval
from lib.datasets import ImbalancedImageFolder
from lib.models import get_backbone_for_dataset, DotClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 10

DATASETS = {"mnist", "cifar10", "fashion_mnist"}


def experiment(
    dataset: str,
    hidden_dim: int = 10,
    random_shift: bool = False,
    verbose: bool = True,
    seed: int = 0,
    random_shift_temperature: float = 0.5,
    **kwargs,
):
    """
    Implementation of Black Box Shift Estimation as introduced in: https://arxiv.org/pdf/1802.03916.pdf
    """
    torch.manual_seed(seed)

    if dataset not in DATASETS:
        raise ValueError(f"Dataset '{dataset}' is not supported")

    params = HYPERPARAMS[dataset]

    epochs = params["epochs"]
    learning_rate = params["learning_rate"]
    batch_size = params["batch_size"]
    weight_decay = params["weight_decay"]
    momentum = params["momentum"]
    num_layers = params["num_laydataseters"]
    transform = params["transform"]

    backbone = get_backbone_for_dataset(dataset=dataset, hidden_dim=hidden_dim)

    model = DotClassifier(backbone, NUM_CLASSES, num_layers=num_layers)
    model.to(DEVICE)

    # Set class distribution
    if random_shift:
        source_label_distribution = label_shift.generate_random_weights(
            NUM_CLASSES, random_shift_temperature
        )
        target_label_distribution = label_shift.generate_random_weights(
            NUM_CLASSES, random_shift_temperature
        )
    else:
        source_label_distribution = label_shift.generate_class_unbalance(NUM_CLASSES)
        target_label_distribution = label_shift.generate_single_class(NUM_CLASSES)

    # Create Datasets
    source_dataset = ImbalancedImageFolder(
        f"data/class_{dataset}/train",
        class_weights=source_label_distribution,
        seed=seed,
        transform=transform,
    )
    train_dataset, val_dataset, _ = random_split(
        source_dataset, [10_000, 1_000, len(source_dataset) - 11_000]
    )
    train_split_1, train_split_2 = random_split(train_dataset, [5_000, 5_000])

    target_dataset = ImbalancedImageFolder(
        f"data/class_{dataset}/test",
        class_weights=target_label_distribution,
        seed=seed,
        transform=transform,
    )
    train_1_dataloader = DataLoader(train_split_1, batch_size=batch_size, num_workers=2)
    train_2_dataloader = DataLoader(train_split_2, batch_size=batch_size, num_workers=2)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)
    target_dataloader = DataLoader(target_dataset, batch_size=batch_size, num_workers=2)

    # Train on first split
    train_history = {"cross_entropy": []}
    val_history = {"cross_entropy": []}
    best_loss = 1e10

    for epoch_idx in range(epochs):
        if verbose:
            print(f"\nEpoch {epoch_idx}")
        _, train_ce_loss, train_accuracy = train(
            model=model,
            dataloader=train_1_dataloader,
            optim=SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            ),
            use_pbar=verbose,
        )
        _, val_ce_loss, val_accuracy = eval(
            model=model,
            dataloader=val_dataloader,
            use_pbar=verbose,
        )
        train_history["cross_entropy"].append(train_ce_loss)
        val_history["cross_entropy"].append(val_ce_loss)

        if val_ce_loss <= best_loss:
            torch.save(model.state_dict(), "./best.pth")
            best_loss = val_ce_loss

    # Calculate confusion matrix on second split
    _, _, _, confusion_matrix_train = eval(
        model=model,
        dataloader=train_2_dataloader,
        use_pbar=verbose,
        return_confusion_matrix=True,
    )
    # Calculate preidiction rates on target
    _, _, _, confusion_matrix_target = eval(
        model=model,
        dataloader=target_dataloader,
        use_pbar=verbose,
        return_confusion_matrix=True,
    )
    mu_target = confusion_matrix_target.sum(axis=1, keepdims=True)  # C x 1
    print("C", confusion_matrix_train)
    print("mu", mu_target)

    eigenvalues, _ = torch.linalg.eig(confusion_matrix_train)
    if eigenvalues.real.min() > 0:
        class_weights = torch.linalg.inv(confusion_matrix_train) @ mu_target
        class_weights = torch.relu(class_weights)
    else:
        class_weights = torch.ones_like(mu_target)
    print("w", class_weights)

    # Retrain on all data with weighted classes
    train_history = {"cross_entropy": []}
    val_history = {"cross_entropy": []}
    best_loss = 1e10

    for epoch_idx in range(epochs):
        if verbose:
            print(f"\nEpoch {epoch_idx}")
        _, train_ce_loss, train_accuracy = train(
            model=model,
            dataloader=train_dataloader,
            optim=SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            ),
            use_pbar=verbose,
            class_weights=class_weights[:, 0],
        )
        _, val_ce_loss, val_accuracy = eval(
            model=model,
            dataloader=val_dataloader,
            use_pbar=verbose,
        )
        train_history["cross_entropy"].append(train_ce_loss)
        val_history["cross_entropy"].append(val_ce_loss)

        if val_accuracy <= best_loss:
            torch.save(model.state_dict(), "./best.pth")
            best_loss = val_accuracy

    # Check accuracy
    _, target_ce_loss, target_accuracy = eval(
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


def multiple_trials(
    experiment_config: Dict, num_trials: int, trial_folder: str
) -> Dict:
    if os.path.isdir(trial_folder):
        shutil.rmtree(trial_folder)
    os.makedirs(trial_folder)

    results = []
    for i in range(num_trials):
        print(f"Experiment {i+1}/{num_trials}")
        trial_results = experiment(**experiment_config, seed=i)
        results.append(trial_results)
        with open(os.path.join(trial_folder, f"trial_{i:03d}.json"), "w") as f:
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
    dataset: str,
    hidden_dim: int = 10,
    num_trials: int = 20,
    spectral_norm: bool = False,
    random_shift: bool = False,
):
    models = [
        {"model_name": "CNN", "cnn": True},
    ]

    if dataset not in DATASETS:
        raise ValueError(f"Dataset '{dataset}' is not supported")

    print(f"Dataset: {dataset}")

    suffix = f"_{dataset}"
    if random_shift:
        suffix += "_random"

    results = []
    for model_config in models:
        experiment_config = {
            "dataset": dataset,
            "spectral_norm": spectral_norm,
            "hidden_dim": hidden_dim,
        }
        experiment_config = {**experiment_config, **model_config}
        exp_results = multiple_trials(
            num_trials=num_trials,
            experiment_config=experiment_config,
            trial_folder=f"trial_results{suffix}_bbse",
        )
        results.append(exp_results)
    results = group_results(results)

    results.to_csv(f"class_imbalance_results{suffix}_bbse.csv", index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment setup")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--random_shift", action="store_true")
    args = parser.parse_args()
    main(dataset=args.dataset, random_shift=args.random_shift)
