# Research question: which clinical characteristics distinguish patients with gallstone
# disease from patients without, and how well can a model predict gallstone disease?
import matplotlib.pyplot as plt
import plotly.express as px   # replaces the two plotly imports at the top
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from load_data import load_clean, load_xlsx, short
import kaleido

COLORS = ["#eb6834", "#2a78d6"]  #orange = gallstones (0), blue = no gallstones (1)


def main():
    # 1. Initial data review
    raw = load_xlsx()
    categorical = raw.columns[raw.nunique() <= 2] #counts how many different values each column has and turns into true/false (keep columns with 5 or less)  )
    continuous = []
    for column in raw.columns:
        if column not in categorical:
            continuous.append(column) #sorted

    print("Shape:", raw.shape) #size of table
    print("Data types:", raw.dtypes.value_counts().to_dict()) #gives raw data type of each columns and how many columns
    print("Missing values:", raw.isna().sum().sum()) #sum of all mising values in table
    print("Duplicate rows:", raw.duplicated().sum()) #sum of all duplicate values in table
    summary = raw[continuous].describe().T #extracts continious table from sheet
    #keep just the lowest, average and highest value
    print(summary[["min", "mean", "max"]].round(1)) #future overview requirement

    #cleaning of extreme values
    too_high = summary["max"] > summary["mean"] + 10 * summary["std"]
    too_low = summary["min"] < summary["mean"] - 10 * summary["std"]
    print("\nFeatures with very extreme values:", list(summary.index[too_high | too_low]))
   
    df, target = load_clean()  # df is cleaned table of measurements, target is diagnosis of each patient (0 or 1)

    #plot 1: class balance
    counts = target.value_counts().sort_index() #count and sort gallstone diagnosis sums so that 0 is the first ros
    plt.bar(["Gallstones", "No gallstones"], counts, color=COLORS) #draws bar chart, with the counts as heights
    plt.title("Class balance")
    plt.savefig("fig_1_class_balance.png", dpi=150)
    plt.close()
    print("Plot 1 succesfull")


    #plot2 : Compare gallstone groups using boxplots
    #find the 6 features most related to the diagnosis
    correlations = df[continuous].corrwith(target)  #continuous columns only: how much each follows the diagnosis (assigns number between -1 to 1, near zero means almost no link)
    strength = correlations.abs()                   #absolute value because of corrwith default function
    strongest = strength.sort_values(ascending=False) #sort the continious columns where strongest correlation are on top
    top = strongest.head(6).index                   #names the 6 features on top of the list

    #build a table with the six features and the group of each patient
    data = df[top].rename(columns=short)            #variable with the 6 names of top list shortened (short def from load_data.py)
    data["Group"] = target.map({0: "Gallstones", 1: "None"})  #creates new column called Group where 0 and 1 are assigned the names "Gallstone" and None

    #step 3: stack the features into one long column: one row per measurement
    data = data.melt(id_vars="Group")               #combines the entirety of the feature colunmns into a single column, so each row is one measurement, must be done beecause px.box limitations

    #step 4: draw one boxplot panel per feature, 3 panels per row
    fig = px.box(data, x="Group", y="value", facet_col="variable", facet_col_wrap=3) #from data, draw boxplots with Group on x-axis and value on y-axis, make one panel for each variable and wrap after 3 panels
    fig.update_yaxes(matches=None)                  #every panel gets its own y-scale
    fig.update_layout(title="Comparison of Features Between Gallstone Groups", showlegend=False) #titling
    fig.show() 
    print ("Plot 2 in browser (downloadable)")

    #plot 3: correlation between the continuous features 
    corr = df[continuous].rename(columns=short).corr()  # correlation between all continuous features, with short names
    plt.figure(figsize=(9, 8)) #draw empty canvas 9x8
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1) #drwa corr table, one square per cell
    plt.colorbar(label="Correlation") 
    plt.xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7) #gives positions for each column, names to write at positions, turn the names so they dont overlap
    plt.yticks(range(len(corr)), corr.columns, fontsize=7) #same for y-axis, obviously don't need rotating here
    plt.tight_layout()
    plt.savefig("fig_3_correlation.png", dpi=150)
    plt.close()
    print("Plot 3 successful")


if __name__ == "__main__":
    main()
