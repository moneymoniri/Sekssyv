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

    # Plot 1: class balance
    counts = target.value_counts().sort_index() #count and sort gallstone diagnosis sums so that 0 is the first ros
    plt.bar(["Gallstones", "No gallstones"], counts, color=COLORS) #draws bar chart, with the counts as heights
    plt.title("Class balance")
    plt.savefig("fig_1_class_balance.png", dpi=150)
    plt.close()
    print("Plot 1 succesfull")


    # Plot 2: Compare gallstone groups using boxplots

    # Find the 6 features most related to the diagnosis
    correlations = df[continuous].corrwith(target) #only the continious columns , corrwith gives how much the feature follows the diagnosis
    top = correlations.abs().sort_values(ascending=False).head(6).index #creates a list of feature names, strongest feature first

    # Table with one row per measurement: Group, Feature, Value
    data = df[top].rename(columns=short) #use only six chosen columns, short name from load_data
    data["Group"] = target.map({0: "Gallstones", 1: "None"}) #replaces diagnosis 0, 1 with a word
    data = data.melt(id_vars="Group")

    # One boxplot panel per feature, 3 panels per row
    fig = px.box(data, x="Group", y="value", facet_col="variable", facet_col_wrap=3)
    fig.update_yaxes(matches=None)  # every panel gets its own y-scale
    fig.update_layout(title="Comparison of Features Between Gallstone Groups", showlegend=False)
    fig.show()

    # Plot 3: correlation between the continuous features (lower triangle only)
    labels = [short(c) for c in continuous]
    corr = df[continuous].corr(method="spearman")
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True  # True for the diagonal and the upper triangle
    corr[mask] = np.nan  # hide those cells
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#f1f1f1")  # color NaNs grey
    plt.figure(figsize=(9, 8))
    plt.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar(label="Spearman correlation")
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    plt.yticks(range(len(labels)), labels, fontsize=7)
    plt.tight_layout()
    plt.savefig("fig_3_correlation.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
