import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from load_data import load_clean


def make_pipeline(model):
    # Scale, keep the k best features, then fit the model. Selecting inside the pipeline
    # means it only sees the training part of each cross-validation fold (no leakage).
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif)),
            ("model", model),
        ]
    )


def main():
    df, target = load_clean()

    X_train, X_test, y_train, y_test = train_test_split(
        df, target, test_size=0.25, random_state=42, stratify=target
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    k = [10, 20, "all"]  # number of features kept

    # Each model with the settings that are tried
    models = {
        "Logistic Regression": (
            make_pipeline(LogisticRegression(max_iter=5000)),
            {"select__k": k, "model__C": [0.01, 0.1, 1, 10]},
        ),
        "Random Forest": (
            make_pipeline(RandomForestClassifier(n_estimators=200, random_state=42)),
            {"select__k": k, "model__max_depth": [4, None], "model__min_samples_leaf": [1, 3]},
        ),
        # A small tree so the decisions are easy to explain
        "Decision Tree": (
            make_pipeline(DecisionTreeClassifier(random_state=42)),
            {"select__k": k, "model__max_depth": [2, 3, 4], "model__min_samples_leaf": [5, 10]},
        ),
    }

    print("0 = gallstones, 1 = no gallstones")

    # Baseline: always predicts the most common class
    baseline = DummyClassifier().fit(X_train, y_train)
    print(f"Baseline test accuracy: {baseline.score(X_test, y_test):.3f}")

    best = {}
    for name, (pipeline, grid) in models.items():
        # Choose the settings with five-fold cross-validation on the training patients
        search = GridSearchCV(pipeline, grid, scoring="roc_auc", cv=cv)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best[name] = model
        prediction = model.predict(X_test)

        print(f"\n{name}")
        print(f"Best settings: {search.best_params_}")
        print(f"Cross-validation AUC: {search.best_score_:.3f}")
        print(f"Train accuracy: {model.score(X_train, y_train):.3f}")
        print(f"Test accuracy: {model.score(X_test, y_test):.3f}")

        # Rows are actual diagnoses and columns are predicted diagnoses
        print("Confusion matrix: class order [0, 1]")
        print(confusion_matrix(y_test, prediction, labels=[0, 1]))

        # Show precision, recall (sensitivity for gallstones) and F1 for both diagnoses
        print(
            classification_report(
                y_test,
                prediction,
                labels=[0, 1],
                target_names=["Gallstones", "No gallstones"],
                digits=3,
                zero_division=0,
            )
        )

        # Evaluate the predicted probability of gallstones
        column = list(model.classes_).index(0)
        probability = model.predict_proba(X_test)[:, column]
        print(f"Gallstone ROC-AUC: {roc_auc_score(y_test == 0, probability):.3f}")

    # Sample output: one model on 10 unseen patients
    sample = X_test.sample(10, random_state=42)
    output = pd.DataFrame(
        {
            "actual": y_test[sample.index].map({0: "gallstones", 1: "none"}),
            "predicted": pd.Series(best["Logistic Regression"].predict(sample), index=sample.index).map({0: "gallstones", 1: "none"}),
        }
    )
    print("\nLogistic Regression on 10 unseen patients:\n", output)

    # Save the tuned models for explain.py
    joblib.dump((best, X_test, y_test), "models.joblib")


if __name__ == "__main__":
    main()
