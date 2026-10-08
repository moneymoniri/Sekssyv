from pathlib import Path

import pandas as pd


DATASET_PATH = Path(__file__).with_name("dataset-uci.xlsx")


def load_xlsx():
    return pd.read_excel(DATASET_PATH)


if __name__ == "__main__":
    df = load_xlsx()
    print(df.shape)
    print(df.iloc[:, 0].value_counts())
