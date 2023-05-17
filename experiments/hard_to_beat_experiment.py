import os
import json
import shutil
from typing import List, Dict

import torch
import numpy as np
import pandas as pd
from torch.optim import SGD
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from abstention.calibration import TempScaling
from abstention.label_shift import EMImbalanceAdapter

from lib.utils import label_shift
from lib.utils.ce_trainer import train, eval
from lib.datasets import ImbalancedImageFolder
from lib.models import get_backbone, DotClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 10


def experiment(
    hidden_dim: int = 10,
    cifar10: bool = False,
    verbose: bool = True,
    seed: int = 0,
    **kwargs,
):
    """
    Implementation of Black Box Shift Estimation as introduced in: https://arxiv.org/pdf/1802.03916.pdf
    """
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

    model = DotClassifier(backbone, NUM_CLASSES, num_layers=1)
    model.to(DEVICE)

    # Create Datasets
    source_dataset = ImbalancedImageFolder(
        f"data/class_{model_name}/train",
        class_weights=label_shift.generate_class_unbalance(NUM_CLASSES),
        seed=seed,
        transform=transform,
    )
    train_dataset, val_dataset, _ = random_split(
        source_dataset, [10_000, 1_000, len(source_dataset) - 11_000]
    )
    target_dataset = ImbalancedImageFolder(
        f"data/class_{model_name}/test",
        class_weights=label_shift.generate_single_class(NUM_CLASSES),
        seed=seed,
        transform=transform,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)
    target_dataloader = DataLoader(target_dataset, batch_size=batch_size, num_workers=2)

    bcts_calibrator_factory = TempScaling(verbose=False, bias_positions="all")
    imbalance_adapter = EMImbalanceAdapter(calibrator_factory=bcts_calibrator_factory)

    # Train on first split
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

    # Calibrate val
    device = next(model.parameters()).device
    valid_preds = []
    valid_labels = []
    with torch.no_grad():
        for inputs, targets in val_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            preds = torch.softmax(logits, dim=-1)

            valid_labels.append(targets.cpu())
            valid_preds.append(preds.cpu())
    valid_labels = torch.cat(valid_labels, dim=0)
    valid_labels = torch.nn.functional.one_hot(
        valid_labels, num_classes=NUM_CLASSES
    ).numpy()
    valid_preds = torch.cat(valid_preds, dim=0).numpy()

    # Adapt target
    target_preds = []
    target_labels = []
    with torch.no_grad():
        for inputs, targets in target_dataloader:
            inputs = inputs.to(device)
            logits = model(inputs)
            preds = torch.softmax(logits, dim=-1)

            target_labels.append(targets.cpu())
            target_preds.append(preds.cpu())
    target_labels = torch.cat(target_labels, dim=0).numpy()
    target_preds = torch.cat(target_preds, dim=0).numpy()

    try:
        imbalance_adapter_func = imbalance_adapter(
            valid_labels=valid_labels,
            tofit_initial_posterior_probs=target_preds,
            valid_posterior_probs=valid_preds,
        )
        adapted_shifted_test_preds = imbalance_adapter_func(target_preds)

        preds = np.argmax(adapted_shifted_test_preds, axis=-1)
        correct = preds == target_labels
        target_accuracy = np.mean(correct)
    except:
        target_accuracy = np.nan

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
    cifar10: bool = False,
    hidden_dim: int = 10,
    num_trials: int = 20,
    spectral_norm: bool = False,
):
    models = [
        {"model_name": "CNN", "cnn": True},
    ]

    if cifar10:
        print("Testing on CIFAR10")
        suffix = "_cifar10"
    else:
        print("Testing on MNIST")
        suffix = "_mnist"

    results = []
    for model_config in models:
        experiment_config = {
            "cifar10": cifar10,
            "spectral_norm": spectral_norm,
            "hidden_dim": hidden_dim,
        }
        experiment_config = {**experiment_config, **model_config}
        exp_results = multiple_trials(
            num_trials=num_trials,
            experiment_config=experiment_config,
            trial_folder=f"trial_results{suffix}_hard_to_beat",
        )
        results.append(exp_results)
    results = group_results(results)

    results.to_csv(f"class_imbalance_results{suffix}_hard_to_beat.csv", index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment setup")
    parser.add_argument("--cifar10", action="store_true")
    args = parser.parse_args()
    main(cifar10=args.cifar10)
