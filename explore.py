# Research question: which clinical characteristics distinguish patients with gallstone
# disease from patients without, and how well can a model predict gallstone disease?
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from load_data import load_clean, load_xlsx, short

COLORS = ["#eb6834", "#2a78d6"]  #orange = gallstones (0), blue = no gallstones (1)


def main():
    # 1. Initial data review
    raw = load_xlsx()
    categorical = raw.columns[raw.nunique() <= 2] #counts how many different values each column has and turns into true/false (keep columns with 2 or less)
    continuous = []
    for column in raw.columns:
        if column not in categorical:
            continuous.append(column) #sorted

    print("Shape:", raw.shape) #size of table
    print("Data types:", raw.dtypes.value_counts().to_dict()) #gives raw data type of each columns and how many columns
    print("Missing values:", raw.isna().sum().sum()) #sum of all mising values in table
    print("Duplicate rows:", raw.duplicated().sum()) #sum of all duplicate values in table
    summary = raw[continuous].describe().T #extracts continious table from sheet #T is for transpose, so that the features are on the left and the statistics are on top
    #keep just the lowest, average and highest value
    print(summary[["min", "mean", "max"]].round(1)) #future overview requirement

    #Showing values that are very extreme, more than 10 standard deviations away from the mean
    too_high = summary["max"] > summary["mean"] + 10 * summary["std"]
    too_low = summary["min"] < summary["mean"] - 10 * summary["std"]
    print("\nFeatures with very extreme values:", list(summary.index[too_high | too_low]))
   
    df, target = load_clean()  # df is cleaned table of measurements, target is diagnosis of each patient (0 or 1)

    #plot 1: class balance
    counts = target.value_counts().sort_index() #count and sort gallstone diagnosis sums so that 0 is the first row
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
    fig = px.box(
        data, x="Group", y="value", color="Group",                 #colour each box by group
        facet_col="variable", facet_col_wrap=3,                     #one panel per feature, 3 per row
        color_discrete_map={"Gallstones": COLORS[0], "None": COLORS[1]},  #same orange/blue as plot 1
        category_orders={"Group": ["Gallstones", "None"]},          #gallstones always on the left
        facet_row_spacing=0.12, facet_col_spacing=0.06,             #more space between panels
        template="plotly_white",                                    #white background instead of grey
    )
    fig.for_each_annotation(lambda a: a.update(text=f"<b>{a.text.split('=')[-1]}</b>", font_size=14))  #"variable=CRP" -> bold "CRP"
    fig.update_yaxes(matches=None, showticklabels=True, title_text="")  #own y-scale and visible numbers on every panel
    fig.update_xaxes(title_text="")                                 #group names under the boxes are enough
    fig.update_traces(boxmean=True, marker=dict(size=4, opacity=0.6), line_width=1.5)  #dashed line = mean, smaller outlier dots
    fig.update_layout(
        title=dict(text="Comparison of Features Between Gallstone Groups", x=0.5, font_size=20),
        boxmode="overlay",                                          #centres each box over its label
        showlegend=False, width=1100, height=700,
        font=dict(family="Arial", size=12),
        margin=dict(t=90, l=40, r=20, b=40),
    )
    fig.show()          
    print("Plot 2 successful, shown in browser")

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
