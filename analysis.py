from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from load_data import load_xlsx


# Use the same model settings in both files
def make_models():
    return {
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
        "Decision Tree": DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=10,
            random_state=42,
        ),
    }


def main():
    df = load_xlsx()
    target = df.pop("Gallstone Status")

    X_train, X_test, y_train, y_test = train_test_split(
        df, target, test_size=0.25, random_state=42, stratify=target
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("0 = gallstones, 1 = no gallstones")

    for name, model in make_models().items():
        # Check accuracy across five training folds
        scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="accuracy"
        )

        # Train the model and predict the test diagnoses
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

        print(f"\n{name}")
        print(f"Cross-validation accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")
        print(f"Test accuracy: {model.score(X_test, y_test):.3f}")

        # Rows are actual diagnoses and columns are predicted diagnoses
        print("Confusion matrix: class order [0, 1]")
        print(confusion_matrix(y_test, prediction, labels=[0, 1]))

        # Show precision, recall and F1 for both diagnoses
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


if __name__ == "__main__":
    main()