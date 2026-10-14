import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

from load_data import load_xlsx
from analysis import make_models


def main():
    df = load_xlsx()
    target = df.pop("Gallstone Status")

    X_train, X_test, y_train, y_test = train_test_split(
        df, target, test_size=0.25, random_state=42, stratify=target
    )

    print("Positive coefficient: towards no gallstones")
    print("Negative coefficient: towards gallstones")
    print("All models below are fitted on training patients only.")

    for name, model in make_models().items():
        model.fit(X_train, y_train)

        # Get coefficients or built-in tree importance
        if name == "Logistic Regression":
            coefficients = model.named_steps["model"].coef_[0]
            ranking = pd.DataFrame(
                {
                    "feature": df.columns,
                    "coefficient": coefficients,
                    "importance": abs(coefficients),
                }
            )
        else:
            ranking = pd.DataFrame(
                {
                    "feature": df.columns,
                    "importance": model.feature_importances_,
                }
            )

        ranking = ranking.sort_values("importance", ascending=False)

        print(f"\n{name}: built-in ranking")
        print(
            ranking[ranking["importance"] > 0]
            .head(10)
            .to_string(index=False)
        )

        # Shuffle each measurement and check the drop in test accuracy
        importance = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="accuracy",
            n_repeats=10,
            random_state=42,
        )

        ranking = pd.DataFrame(
            {
                "feature": df.columns,
                "accuracy_drop": importance.importances_mean,
                "shuffle_sd": importance.importances_std,
            }
        ).sort_values("accuracy_drop", ascending=False)

        print(f"\n{name}: held-out permutation importance")
        print(ranking.head(10).to_string(index=False))


if __name__ == "__main__":
    main()