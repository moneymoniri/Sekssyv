import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from load_data import load_xlsx


def main():
    df = load_xlsx()
    target = df.pop("Gallstone Status")

    logistic_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000)),
        ]
    )

    random_forest = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    decision_tree = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=10,
        random_state=42,
    )

    # Train Logistic Regression
    logistic_model.fit(df, target)

    # Get the coefficient for each measurement
    logistic_coefficients = logistic_model.named_steps["model"].coef_[0]

    logistic_result = pd.DataFrame(
        {
            "feature": df.columns,
            "coefficient": logistic_coefficients,
        }
    )

    # Rank by size, regardless of whether the coefficient is positive or negative
    logistic_result["importance"] = logistic_result["coefficient"].abs()
    logistic_result = logistic_result.sort_values(
        "importance", ascending=False
    )

    print("\nLogistic Regression: top 10 measurements")
    print(logistic_result.head(10).to_string(index=False))

    # Train Random Forest
    random_forest.fit(df, target)

    # Get the importance of each measurement
    random_forest_result = pd.DataFrame(
        {
            "feature": df.columns,
            "importance": random_forest.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    print("\nRandom Forest: top 10 measurements")
    print(random_forest_result.head(10).to_string(index=False))

    # Train Decision Tree
    decision_tree.fit(df, target)

    # Get the importance of each measurement
    decision_tree_result = pd.DataFrame(
        {
            "feature": df.columns,
            "importance": decision_tree.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    print("\nDecision Tree: top 10 measurements")
    print(decision_tree_result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()