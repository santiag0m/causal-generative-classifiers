import json
from pathlib import Path
from typing import List, Dict, Tuple

import matplotlib
import matplotlib.pyplot as plt

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


def main(trial_folder: str):
    matplotlib.rcParams.update({"font.size": 16})

    trial_results = load_trials(trial_folder)
    plt.ion()
    f, ax = plt.subplots()

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

    ax.set_xlim([0.9 * hsic_min, 1.1 * hsic_max])
    ax.set_ylim([0.9 * ce_min, 1.1 * ce_max])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.minorticks_on()
    ax.grid(which="both")
    ax.set_ylabel("Cross Entropy")
    ax.set_xlabel("HSIC")
    ax.set_title("Training Trajectory")
    plt.show()
