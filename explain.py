import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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

    logistic_model.fit(df, target)
    logistic_coefficients = logistic_model.named_steps["model"].coef_[0]
    logistic_result = pd.DataFrame(
        {
            "feature": df.columns,
            "coefficient": logistic_coefficients,
        }
    ).sort_values("coefficient")
    print("\nLogistic Regression coefficients:")
    print(logistic_result.to_string(index=False))

    random_forest.fit(df, target)
    random_forest_result = pd.DataFrame(
        {
            "feature": df.columns,
            "importance": random_forest.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    print("\nRandom Forest feature importance:")
    print(random_forest_result.to_string(index=False))


if __name__ == "__main__":
    main()
