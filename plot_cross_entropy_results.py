import matplotlib.pyplot as plt
import pandas as pd


def format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    train_df = df.query("model_name == 'CNN_train'").set_index("exp")[["Accuracy"]]
    target_df = df.query("model_name == 'CNN_target'").set_index("exp")[["Accuracy"]]
    df = train_df.join(target_df, lsuffix="_Train", rsuffix="_Target").reset_index()
    return df


if __name__ == "__main__":
    df_control = format_dataframe(pd.read_csv("ce_results.csv"))
    df_residual = format_dataframe(pd.read_csv("ce_results_residual.csv"))

    plt.ion()
    f, ax = plt.subplots()
    ax.scatter(df_control["Accuracy_Train"], df_control["Accuracy_Target"], c="blue")
    ax.scatter(df_residual["Accuracy_Train"], df_residual["Accuracy_Target"], c="green")
