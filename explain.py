import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.tree import export_text

from load_data import short


def main():
    models, X_test, y_test = joblib.load("models.joblib")

    print("Models were tuned and fitted on training patients only.")

    # Shuffle each measurement on the unseen test patients and check the drop in AUC.
    # A big drop means the model relies on that measurement.
    importance = pd.DataFrame(
        {
            name: permutation_importance(
                model, X_test, y_test, scoring="roc_auc", n_repeats=30, random_state=42
            ).importances_mean
            for name, model in models.items()
        },
        index=[short(c) for c in X_test.columns],
    )

    # Rank within each model and average: features high in all models are most trustworthy
    importance["mean_rank"] = importance.rank(ascending=False).mean(axis=1)
    importance = importance.sort_values("mean_rank")
    print("\nTop 10 measurements (drop in AUC per model):")
    print(importance.head(10).round(3))
    importance.round(3).to_csv("importance.csv")

    importance.head(10).drop(columns="mean_rank").iloc[::-1].plot.barh(figsize=(8, 6))
    plt.xlabel("Drop in AUC when the measurement is shuffled")
    plt.tight_layout()
    plt.savefig("fig_4_importance.png", dpi=150)

    # Direction of the effects (the measurements are standardized)
    print("\nPositive coefficient: towards no gallstones")
    print("Negative coefficient: towards gallstones")
    logistic = models["Logistic Regression"]
    kept = X_test.columns[logistic.named_steps["select"].get_support()]
    coefficients = pd.Series(logistic.named_steps["model"].coef_[0], index=[short(c) for c in kept])
    print(coefficients.sort_values(key=abs, ascending=False).head(10).round(2))

    # The small decision tree written out as rules
    tree = models["Decision Tree"]
    kept = X_test.columns[tree.named_steps["select"].get_support()]
    print("\nDecision Tree rules (0 = gallstones, values are standardized):")
    print(export_text(tree.named_steps["model"], feature_names=[short(c) for c in kept]))


if __name__ == "__main__":
    main()
