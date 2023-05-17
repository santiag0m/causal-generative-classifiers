import os
import json
import shutil
from typing import List, Dict

import torch
import pandas as pd
from torch.optim import SGD
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from lib.utils import label_shift
from lib.utils.mdmm import MDMMOptim
from lib.utils.mdmm_trainer import train, eval
from lib.datasets import ImbalancedImageFolder
from lib.models import get_backbone, CGCResidual, DotClassifier
from lib.utils.expectation_maximization import expectation_maximization

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 10


def experiment(
    hidden_dim: int = 10,
    cifar10: bool = False,
    spectral_norm: bool = False,
    use_hsic: bool = False,
    verbose: bool = True,
    seed: int = 0,
    use_residual: bool = False,
    random_shift: bool = False,
    random_shift_temperature: float = 0.5,
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

    print("Source Distribution: ", source_label_distribution)
    print("Target distribution: ", target_label_distribution)

    # Create Datasets
    source_dataset = ImbalancedImageFolder(
        f"data/class_{model_name}/train",
        class_weights=source_label_distribution,
        seed=seed,
        transform=transform,
    )
    train_dataset, val_dataset, _ = random_split(
        source_dataset, [10_000, 1_000, len(source_dataset) - 11_000]
    )
    target_dataset = ImbalancedImageFolder(
        f"data/class_{model_name}/test",
        class_weights=target_label_distribution,
        seed=seed,
        transform=transform,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)
    target_dataloader = DataLoader(target_dataset, batch_size=batch_size, num_workers=2)

    # Setup Optimizer
    mmdm_optim = MDMMOptim(
        params=model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        model_optim=torch.optim.SGD,
    )

    if use_residual:
        # Fit class priors before training
        model.fit_class_probs(train_dataloader)

    # Train
    train_history = {"cross_entropy": [], "hsic": []}
    val_history = {"cross_entropy": [], "hsic": []}
    best_loss = 1e10

    for epoch_idx in range(epochs):
        if verbose:
            print(f"\nEpoch {epoch_idx}")
        train_hsic, train_ce_loss, train_accuracy = train(
            model=model,
            dataloader=train_dataloader,
            optim=mmdm_optim,
            use_pbar=verbose,
            use_hsic=use_hsic,
        )
        val_hsic, val_ce_loss, val_accuracy = eval(
            model=model,
            dataloader=val_dataloader,
            use_pbar=verbose,
        )
        train_history["cross_entropy"].append(train_ce_loss)
        train_history["hsic"].append(train_hsic)
        val_history["cross_entropy"].append(val_ce_loss)
        val_history["hsic"].append(val_hsic)

        if val_accuracy <= best_loss:
            torch.save(model.state_dict(), "./best.pth")
            best_loss = val_accuracy

    # Adjust marginal
    if use_residual:
        _, y_marginal = expectation_maximization(model, target_dataloader)
        print("True marginal: ", target_label_distribution)
        print("Estimated marginal: ", y_marginal)
        model.class_probs.copy_(y_marginal)

    # Check accuracy
    target_hsic, target_ce_loss, target_accuracy = eval(
        model=model,
        dataloader=target_dataloader,
        use_pbar=verbose,
    )

    print(
        f"{train_accuracy=:.4f}, {val_accuracy=:.4f}, {target_accuracy=:.4f}, {target_hsic:=.4f}\n"
    )

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
    use_residual: bool = False,
    use_hsic: bool = False,
    spectral_norm: bool = False,
    random_shift: bool = False,
    trial_folder_prefix: str = "trial_results",
):
    models = [
        {"model_name": "CNN", "cnn": True},
    ]

    suffix = ""
    if cifar10:
        suffix += "_cifar10"
    else:
        suffix += "_mnist"

    if random_shift:
        suffix += "_random"
    if use_residual:
        suffix += "_residual"
    if use_hsic:
        suffix += "_hsic"
    if spectral_norm:
        suffix += "_sn"

    results = []
    for model_config in models:
        experiment_config = {
            "cifar10": cifar10,
            "use_residual": use_residual,
            "use_hsic": use_hsic,
            "spectral_norm": spectral_norm,
            "random_shift": random_shift,
            "hidden_dim": hidden_dim,
        }
        experiment_config = {**experiment_config, **model_config}
        exp_results = multiple_trials(
            num_trials=num_trials,
            experiment_config=experiment_config,
            trial_folder=trial_folder_prefix + suffix,
        )
        results.append(exp_results)
    results = group_results(results)

    results.to_csv(f"class_imbalance_results{suffix}.csv", index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment setup")
    parser.add_argument("--use_residual", action="store_true")
    parser.add_argument("--use_hsic", action="store_true")
    parser.add_argument("--spectral_norm", action="store_true")
    parser.add_argument("--cifar10", action="store_true")
    parser.add_argument("--random_shift", action="store_true")
    args = parser.parse_args()
    main(
        cifar10=args.cifar10,
        use_residual=args.use_residual,
        use_hsic=args.use_hsic,
        spectral_norm=args.spectral_norm,
        random_shift=args.random_shift,
    )
