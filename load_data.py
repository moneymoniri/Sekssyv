import re
from pathlib import Path
import pandas as pd


DATASET_PATH = Path(__file__).with_name("dataset-uci.xlsx")


def load_xlsx():
    return pd.read_excel(DATASET_PATH)


def short(name):
    # "Lean Mass (LM) (%)" -> "LM" (used as label in the figures)
    match = re.search(r"\((\w[\w/]*)\)", name)
    return match.group(1) if match else name


def load_clean():
    df = load_xlsx()

    # Obesity (%) = 1954 is a typing error (the next largest value is 126): use the median
    df.loc[df["Obesity (%)"] > 500, "Obesity (%)"] = df["Obesity (%)"].median()

    target = df.pop("Gallstone Status")
    return df, target


if __name__ == "__main__":
    df = load_xlsx()
    print(df.shape)
    print(df["Gallstone Status"].value_counts())
