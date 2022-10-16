import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns


def format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    train_df = df.query("setting == 'train'")[["exp", "Accuracy", "setting"]]
    val_df = df.query("setting == 'val'")[["exp", "Accuracy", "setting"]]
    target_df = df.query("setting == 'target'")[["exp", "Accuracy", "setting"]]
    df = pd.concat([train_df, val_df, target_df], axis=0).reset_index()
    return df


if __name__ == "__main__":
    df_control = format_dataframe(pd.read_csv("cifar10_ce_results.csv"))
    df_residual = format_dataframe(pd.read_csv("cifar10_ce_results_residual.csv"))

    plt.ion()
    df_control["model"] = "dot product"
    df_residual["model"] = "residual + CGC"
    df = pd.concat([df_control, df_residual], axis=0)

    sns.boxplot(data=df, x="Accuracy", y="setting", hue="model")
