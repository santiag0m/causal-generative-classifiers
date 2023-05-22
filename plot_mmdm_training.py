import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lib.plot.colorline import colorline


def load_trial_json(filepath: str) -> Dict:
    with open(filepath, "r") as f:
        return json.load(f)


def load_trials(trial_folder: str) -> List[Dict]:
    trial_files = Path(trial_folder).glob("trial_*.json")
    trial_results = [load_trial_json(filename) for filename in trial_files]
    return trial_results


def update_min_max(
    min_val: float, max_val: float, data: List[Dict]
) -> Tuple[float, float]:
    min_data = min(data)
    max_data = max(data)
    if min_val > min_data:
        min_val = min_data
    if max_val < max_data:
        max_val = max_data
    return min_val, max_val


def main(
    trial_folder: str, ax: Optional[plt.Axes] = None, f: Optional[plt.Figure] = None
):
    matplotlib.rcParams.update({"font.size": 16})

    trial_results = load_trials(trial_folder)
    plt.ion()

    if (not f) and (not ax):
        f, ax = plt.subplots()
    elif not ax:
        ax = plt.subplots()

    hsic_min = 1
    hsic_max = 0

    ce_min = 1_000_000
    ce_max = 0

    for trial in trial_results:
        hsic_loss = trial["train_history"]["hsic"]
        ce_loss = trial["train_history"]["cross_entropy"]
        colorline(hsic_loss, ce_loss, ax=ax, alpha=0.3)
        hsic_min, hsic_max = update_min_max(hsic_min, hsic_max, hsic_loss)
        ce_min, ce_max = update_min_max(ce_min, ce_max, ce_loss)

    if f is not None:
        # Create a ScalarMappable with your desired cmap and normalization
        sm = cm.ScalarMappable(
            cmap=plt.get_cmap("coolwarm_r"), norm=plt.Normalize(0.0, 1.0)
        )

        # Create a divider for the existing axes instance
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        # Plot the colorbar with custom tick labels
        cbar = f.colorbar(sm, cax=cax, ticks=[0, 1], orientation="vertical")
        cbar.ax.set_yticklabels(["start", "finish"])

    ax.set_xlim([0.9 * hsic_min, 1.1 * hsic_max])
    ax.set_ylim([0.9 * ce_min, 1.1 * ce_max])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.minorticks_on()
    ax.grid(which="both")
    ax.set_ylabel("Cross Entropy")
    ax.set_xlabel("HSIC")
    ax.set_title("Training Trajectory")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plt.ion()

    mnist_figure, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    main("trial_results_mnist_residual_sn//", ax=axs[0])
    main("trial_results_mnist_residual_hsic_sn//", f=mnist_figure, ax=axs[1])
    axs[0].set_title("Unconstrained")
    axs[1].set_title("HSIC Constrained")
    mnist_figure.suptitle("MNIST Training")
    mnist_figure.subplots_adjust(wspace=0.15)
    axs[0].set_xlim([0.024, 0.062])

    cifar10_figure, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    main("trial_results_cifar10_residual_sn//", ax=axs[0])
    main("trial_results_cifar10_residual_hsic_sn//", f=cifar10_figure, ax=axs[1])
    axs[0].set_title("Unconstrained")
    axs[1].set_title("HSIC Constrained")
    cifar10_figure.suptitle("CIFAR10 Training")
    cifar10_figure.subplots_adjust(wspace=0.15)
