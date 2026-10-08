from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from load_data import load_xlsx


def main():
    df = load_xlsx()
    target = df.pop("Gallstone Status")
    X_train, X_test, y_train, y_test = train_test_split(
        df, target, test_size=0.25, random_state=42, stratify=target
    )
    models = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000)),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        print(f"{name} accuracy: {model.score(X_test, y_test):.3f}")
        print(f"{name} prediction: {prediction.tolist()}")


if __name__ == "__main__":
    main()
