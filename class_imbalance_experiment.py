import os
import json
import shutil
from typing import List, Dict

import torch
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from lib.utils.mmdm import MMDMOptim
from lib.utils.mmdm_trainer import train, eval
from lib.datasets import ImbalancedImageFolder
from lib.models import get_backbone, CGCResidual
from lib.utils.expectation_maximization import expectation_maximization

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 10

TRIAL_FOLDER = "trial_results"

transform = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)


def generate_uniform_weights():
    return {str(i): 1.0 / NUM_CLASSES for i in range(NUM_CLASSES)}


def generate_random_weights():
    weights = torch.softmax(torch.randn(NUM_CLASSES), 0)
    return {str(i): weights[i] for i in range(NUM_CLASSES)}


def generate_class_unbalance():
    weights = torch.softmax(
        torch.tensor([2, 2, 2, 2, 2, 1, 1, 1, 1, 1], dtype=torch.float32), 0
    )
    return {str(i): weights[i] for i in range(NUM_CLASSES)}


def experiment(
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    cnn: bool = False,
    mlp_layers: List[int] = [],
    spectral_norm: bool = False,
    only_cross_entropy: bool = False,
    verbose: bool = True,
    seed: int = 0,
    **kwargs,
):
    torch.manual_seed(seed)
    backbone = get_backbone(cnn=cnn, mlp_layers=mlp_layers, spectral_norm=spectral_norm)
    model = CGCResidual(
        backbone, NUM_CLASSES, spectral_norm=spectral_norm, num_layers=1
    )
    model.to(DEVICE)

    # Create Datasets
    source_dataset = ImbalancedImageFolder(
        "data/class_mnist/train",
        class_weights=generate_class_unbalance(),
        seed=seed,
        transform=transform,
    )
    train_dataset, val_dataset, _ = random_split(
        source_dataset, [10_000, 1_000, len(source_dataset) - 11_000]
    )
    target_dataset = ImbalancedImageFolder(
        "data/class_mnist/test",
        class_weights=generate_uniform_weights(),
        seed=seed,
        transform=transform,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    target_dataloader = DataLoader(target_dataset, batch_size=batch_size)

    # Setup Optimizer
    mmdm_optim = MMDMOptim(
        params=model.parameters(), lr=learning_rate, model_optim=torch.optim.SGD
    )

    # Fit class priors before training
    model.fit_class_probs(train_dataloader)

    # Train
    train_history = {"hsic": [], "cross_entropy": []}
    val_history = {"hsic": [], "cross_entropy": []}
    best_loss = 1e10

    for epoch_idx in range(num_epochs):
        if verbose:
            print(f"\nEpoch {epoch_idx}")
        train_hsic_loss, train_ce_loss, train_accuracy = train(
            model=model,
            dataloader=train_dataloader,
            mmdm_optim=mmdm_optim,
            use_pbar=verbose,
            use_hsic=not only_cross_entropy,
        )
        val_hsic_loss, val_ce_loss, val_accuracy = eval(
            model=model,
            dataloader=val_dataloader,
            use_pbar=verbose,
            use_hsic=not only_cross_entropy,
        )
        train_history["hsic"].append(train_hsic_loss)
        val_history["hsic"].append(val_hsic_loss)
        train_history["cross_entropy"].append(train_ce_loss)
        val_history["cross_entropy"].append(val_ce_loss)

        if val_accuracy <= best_loss:
            torch.save(model.state_dict(), "./best.pth")
            best_loss = val_accuracy

    # Adjust marginal
    if not only_cross_entropy:
        _, y_marginal = expectation_maximization(model, target_dataloader)
        print(y_marginal)
        model.class_probs.copy_(y_marginal)

    # Check accuracy
    target_hsic_loss, target_ce_loss, target_accuracy = eval(
        model=model,
        dataloader=target_dataloader,
        use_pbar=verbose,
        use_hsic=True,
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
        # "val": pd.Series(val_accuracy).fillna(0).rename(experiment_config["model_name"]),
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
        df["model_name"] = df["model_name"].apply(lambda x: x + f"_{key}")
        df_list.append(df)
    df = pd.concat(df_list)

    return df


def plot_results(df: pd.DataFrame, title: str = ""):
    plt.ion()

    ax = sns.boxplot(x="Accuracy", y="model_name", hue="loss_criterion", data=df)
    ax.set(xscale="log")
    ax.set_title(title)
    ax.set_xscale("linear")


def main(
    num_trials: int = 20,
    num_epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 5e-2,
    spectral_norm: bool = False,
    only_cross_entropy: bool = False,
):
    models = [
        {"model_name": "CNN", "cnn": True},
        # {"model_name": "MLP 2x256", "mlp_layers": [256, 256, 10]},
        # {"model_name": "MLP 2x524", "mlp_layers": [524, 524, 10]},
        # {"model_name": "MLP 2x1024", "mlp_layers": [1024, 1024, 10]},
        # {"model_name": "MLP 4x256", "mlp_layers": [256, 256, 256, 256, 10]},
        # {"model_name": "MLP 4x524", "mlp_layers": [524, 524, 524, 524, 10]},
        # {"model_name": "MLP 4x1024", "mlp_layers": [1024, 1024, 1024, 1024, 10]},
    ]

    results = []
    for model_config in models:
        experiment_config = {
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "spectral_norm": spectral_norm,
            "only_cross_entropy": only_cross_entropy,
        }
        experiment_config = {**experiment_config, **model_config}
        exp_results = multiple_trials(
            num_trials=num_trials, experiment_config=experiment_config
        )
        results.append(exp_results)
    results = group_results(results)
    results["loss_criterion"] = "HSIC Classification"
    if only_cross_entropy:
        results.to_csv("mmdm_results_ce.csv", index=False)
    else:
        results.to_csv("mmdm_results.csv", index=False)
    plot_results(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment setup")
    parser.add_argument("--only_cross_entropy", action="store_true")
    parser.add_argument("--spectral_norm", action="store_true")
    args = parser.parse_args()
    main(only_cross_entropy=args.only_cross_entropy, spectral_norm=args.spectral_norm)
