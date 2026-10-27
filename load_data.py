import re
from pathlib import Path
import pandas as pd

# Problem Definition:
# The goal of this project is to see if we can use machine learning to 
# predict whether a patient has gallstones based on medical information
# such as age, BMI, cholesterol levels, and other health measurements.
# We compare three models: Logistic Regression, Random Forest, and
# Decision Tree, to see which one makes the most accurate predictions 
# and use different settings for the models to optimize their performance.

DATASET_PATH = Path(__file__).with_name("dataset-uci.xlsx") 


def load_xlsx():
    return pd.read_excel(DATASET_PATH) #Reads excel sheet into pandas table,
#one row per patient and one column per measurement


def short(name):
    # Use the abbreviation in brackets: "Lean Mass (LM) (%)" -> "LM"
    if "(" not in name:
        return name  # no abbreviation, e.g. "Age"
    abbreviation = name.split("(")[1].split(")")[0]
    if abbreviation == "%":
        return name  # "Obesity (%)" has no real abbreviation
    return abbreviation

def load_clean():
    gallstone = load_xlsx()

    #drop rows with null values, there are none in the dataset but just to make sure
    gallstone = gallstone.dropna(how="any")

    #drop duclicated rows and reindex and fixes index
    gallstone.drop_duplicates(inplace=True)
    gallstone.reset_index(drop=True, inplace=True)

    #Obesity (%) = 1954 is a data entry error (the next largest value is 126)
    mask = gallstone["Obesity (%)"] > 500
    gallstone.loc[mask, "Obesity (%)"] = gallstone["Obesity (%)"].median()
    #rows where mask is True in the column Obesity (%)

    #split the measurement from diagnosis, since the MLM´s are supposed to predict the diagnosis
    target = gallstone["Gallstone Status"]
    df = gallstone.drop(columns="Gallstone Status")
    return df, target


if __name__ == "__main__":
    df = load_xlsx()
    print(df.shape)
    print(df["Gallstone Status"].value_counts()) #just a check
