# Research question: which clinical characteristics distinguish patients with gallstone
# disease from patients without, and how well can a model predict gallstone disease?
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from load_data import load_clean, load_xlsx, short

COLORS = ["#eb6834", "#2a78d6"]  # orange = gallstones (0), blue = no gallstones (1)


def main():
    # 1. Initial data review
    raw = load_xlsx()
    categorical = raw.columns[raw.nunique() <= 5]
    continuous = [c for c in raw.columns if c not in categorical]

    print("Shape:", raw.shape)
    print("Data types:", raw.dtypes.value_counts().to_dict())
    print("Missing values:", raw.isna().sum().sum())
    print("Duplicate rows:", raw.duplicated().sum())
    print("Categorical features:", list(categorical))
    print(raw[continuous].describe().T[["min", "mean", "max"]].round(1))

    # 2. Cleaning: look for extreme values (further than 3 x IQR from the quartiles)
    q1, q3 = raw[continuous].quantile(0.25), raw[continuous].quantile(0.75)
    extreme = (raw[continuous] < q1 - 3 * (q3 - q1)) | (raw[continuous] > q3 + 3 * (q3 - q1))
    print("\nExtreme values per feature:", extreme.sum()[extreme.sum() > 0].to_dict())
    print("Largest Obesity (%):", raw["Obesity (%)"].nlargest(3).tolist(), "-> 1954 is a typing error")
    print("The other extreme values (high glucose, CRP) are plausible, so they are kept.")

    df, target = load_clean()  # cleaned data, 0 = gallstones, 1 = no gallstones

    # Plot 1: class balance
    counts = target.value_counts().sort_index()
    plt.bar(["Gallstones", "No gallstones"], counts, color=COLORS)
    plt.title("Class balance")
    plt.savefig("fig_1_class_balance.png", dpi=150)
    plt.close()
    print(f"\nPlot 1: {counts[0]} vs {counts[1]} patients, so the classes are balanced.")

    # Plot 2: the six features that differ most between the groups (Mann-Whitney test)
    p = {c: mannwhitneyu(df.loc[target == 0, c], df.loc[target == 1, c]).pvalue for c in df.columns}
    top = sorted(p, key=p.get)[:6]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, feature in zip(axes.ravel(), top):
        box = ax.boxplot([df.loc[target == 0, feature], df.loc[target == 1, feature]], patch_artist=True)
        for patch, color in zip(box["boxes"], COLORS):
            patch.set_facecolor(color)
        ax.set_xticks([1, 2], ["Gallstones", "None"])
        ax.set_title(f"{short(feature)} (p={p[feature]:.0e})", fontsize=9)
    plt.tight_layout()
    plt.savefig("fig_2_group_differences.png", dpi=150)
    plt.close()
    print("Plot 2: the groups differ most in", [short(f) for f in top])

    # Plot 3: correlation between the continuous features
    labels = [short(c) for c in continuous]
    plt.figure(figsize=(9, 8))
    plt.imshow(df[continuous].corr(method="spearman"), cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Spearman correlation")
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    plt.yticks(range(len(labels)), labels, fontsize=7)
    plt.tight_layout()
    plt.savefig("fig_3_correlation.png", dpi=150)
    plt.close()
    print("Plot 3: the body-composition features (weight, BMI, water, fat, muscle) are strongly")
    print("correlated, so they share importance when we explain the models.")


if __name__ == "__main__":
    main()
