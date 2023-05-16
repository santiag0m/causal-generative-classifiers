import os
import pandas as pd


def tabulate(csv_results: str, prefix: str = "") -> pd.Series:
    df = pd.read_csv(csv_results)

    low_accuracy: pd.Series = (
        df.groupby("setting")["Accuracy"].quantile(0.05).rename("low")
    )
    median_accuracy: pd.Series = (
        df.groupby("setting")["Accuracy"].quantile(0.5).rename("median")
    )
    high_accuracy: pd.Series = (
        df.groupby("setting")["Accuracy"].quantile(0.95).rename("high")
    )

    df = pd.concat([low_accuracy, median_accuracy, high_accuracy], axis=1)
    df = df.applymap(lambda x: f"{x:.3f}")

    accuracy = df.apply(lambda x: f"{x[1]} ({x[0]} - {x[2]})", axis=1)

    label = csv_results.replace(".csv", "")
    label = label.replace(prefix, "")
    label = label.strip("_")

    if not label:
        label = "baseline"

    accuracy = accuracy.rename(label)

    return accuracy


def is_results_file(filename: str, prefix: str = "") -> bool:
    is_csv = filename.endswith(".csv")
    has_prefix = filename.startswith(prefix)
    return is_csv and has_prefix


def main(
    results_folder: str, prefix: str = "class_imbalance_results_mnist"
) -> pd.DataFrame:
    files = [
        file for file in os.listdir(results_folder) if is_results_file(file, prefix)
    ]
    accuracy = [tabulate(file, prefix) for file in sorted(files)]
    df = pd.concat(accuracy, axis=1)

    df = df.T

    return df
