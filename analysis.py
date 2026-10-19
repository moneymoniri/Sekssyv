import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from load_data import load_clean

model_reasons = {
    "Logistic Regression":
        "A simple baseline for binary classification. "
        "It is useful for understanding how patient features "
        "are associated with gallstone status.",

    "Random Forest":
        "Can capture complex and non-linear relationships "
        "between patient measurements and gallstone status.",

    "Decision Tree":
        "Makes predictions using simple rules, making "
        "the results easy to interpret."
    }


def make_pipeline(model):
    # Pipeline to scale the measurements and then fit the model. The scaling is important because the measurements are on different scales, and we want to avoid biasing the model towards measurements with larger values.
    return Pipeline( 
        [
            ("scaler", MinMaxScaler()), # makes all values to values between 0 and 1, so the models are not biased by the scale of the measurements
            ("select", SelectKBest(score_func=f_classif, k="all")),
            ("model", model), # defines the model to be used
        ]
    )


def main():
    X, y = load_clean()

    # 2. Split the data into 75% training and 25% testing use random_state=42 to ensure reproducibility in the training and testing split. This means that the same patients will be in the training and testing sets each time the code is run.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # 3. Use 5-fold cross-validation when training the models
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 4. Choose the models and settings we want to test
    models = {
        "Logistic Regression": (
            make_pipeline(LogisticRegression(max_iter=5000)), # make 5000 iterations to ensure convergence and use pipeline to scale the data before fitting the model
            {"model__C": [0.01, 0.1, 1, 10]} #Model C decides how much we want to regularize the model. A smaller C means more regularization, which can help prevent overfitting. We will test different values of C to find the best one for our data.
        ),

        "Random Forest": (
            make_pipeline(RandomForestClassifier(n_estimators=200, random_state=42)), # creates 200 random forest branches to ensure the model is robust and use pipeline to scale the data before fitting the model. Use random_state=42 to ensure reproducibility in the random forest model. This means that the same random numbers will be generated each time the code is run, which can help with debugging and comparing results.
            {
                "model__max_depth": [4, None], # Maximum depth of the tree to prevent overfitting. A smaller max_depth means the model will be simpler and less likely to overfit the training data. We will test different values of max_depth to find the best one for our data.
                "model__min_samples_leaf": [1, 3] # Minimum number of samples required to be at a leaf node, test both 1 and 3 to find the best one for our data.
            }
        ),

        "Decision Tree": (
            make_pipeline(DecisionTreeClassifier(random_state=42)),
            {
                "model__max_depth": [2, 3, 4], # Maximum depth of the tree to prevent overfitting. A smaller max_depth means the model will be simpler and less likely to overfit the training data. We will test different values of max_depth to find the best one for our data.
                "model__min_samples_leaf": [5, 10] # Minimum number of samples required to be at a leaf node, test both 5 and 10 to find the best one for our data. Use a bit larger min_samples_leaf to prevent overfitting, as decision trees can easily overfit the training data if they are allowed to grow too deep or have too few samples at the leaves.
            }
        )
    }

    print("0 = Gallstones, 1 = No gallstones")

    # 5. Create a simple baseline for comparison. The dummy classifier will always predict the most common class in the training data. This is a simple baseline to compare our models against, to see if they are actually learning anything useful from the data.
    baseline = DummyClassifier()
    baseline.fit(X_train, y_train)

    print("Baseline accuracy:", round(baseline.score(X_test, y_test), 3)) # prints the baseline accuracy on the test data, rounded to 3 decimal places. This is the accuracy of the dummy classifier, which always predicts the most common class in the training data. We will use this as a baseline to compare our models against, to see if they are actually learning anything useful from the data.

    # 6. Train all three models and find their best settings
    best = {}

    for name in models:  
        pipeline = models[name][0] #choose the pipeline for the model, which includes scaling the data and fitting the model. The pipeline is defined in the models dictionary, which contains the model name as the key and a tuple of the pipeline and settings as the value.
        settings = models[name][1] #choose the settings for the model, which are defined in the models dictionary. The settings are a dictionary of hyperparameters to test for the model, which will be used in a grid search to find the best combination of hyperparameters for the model.

        # Try different settings using 5-fold cross-validation
        search = GridSearchCV( #test different settings for the model using a grid search with cross-validation. The grid search will try all combinations of the hyperparameters defined in the settings dictionary, and will evaluate each combination using 5-fold cross-validation. The best combination of hyperparameters will be selected based on the ROC-AUC score, which is a measure of how well the model separates the two classes (gallstones vs no gallstones).
            pipeline,
            settings,
            scoring="roc_auc", # tells the grid search to use the ROC-AUC score to evaluate the models. The ROC-AUC score is a measure of how well the model separates the two classes (gallstones vs no gallstones). A higher ROC-AUC score means the model is better at separating the two classes.
            cv=cv # used to split the training data into 5 folds for cross-validation. The cv parameter is set to the StratifiedKFold object defined earlier, which ensures that each fold has a similar distribution of the target variable (gallstones vs no gallstones). This helps to prevent overfitting and ensures that the model is evaluated on different subsets of the training data.
        )

        # Train using the training data
        search.fit(X_train, y_train)

        # Get the best version of the model
        model = search.best_estimator_

        # Save the trained model
        best[name] = model

        # Predict whether the test patients have gallstones
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 0]

        # 7. Print the results
        print("\n", name)

        print("Best settings:", search.best_params_)
        print("Cross-validation AUC:", round(search.best_score_, 3))
        print("Training accuracy:", round(model.score(X_train, y_train), 3))
        print("Test accuracy:", round(model.score(X_test, y_test), 3))

        # Compare the predictions with the correct diagnoses
        print("Confusion matrix:")
        print(confusion_matrix(y_test, predictions, labels=[0, 1]))

        # Show precision and recall
        print("Gallstone recall:", round(recall_score(y_test, predictions, pos_label=0), 3)) # Show the how effectively the model identifies patients with gallstones. The recall score is the proportion of true positives (patients with gallstones) that are correctly identified by the model. A higher recall score means the model is better at identifying patients with gallstones.

        auc = roc_auc_score(y_test == 0, probabilities)
        print("Gallstone ROC-AUC:", round(auc, 3)) # calculates the ROC-AUC score for the model's predictions on the test data, specifically for the class of patients with gallstones (label 0). The ROC-AUC score is a measure of how well the model separates the two classes (gallstones vs no gallstones). A higher ROC-AUC score means the model is better at separating the two classes.

    # 8. Test Logistic Regression on 10 patients (Test on unseen data)
    sample = X_test.sample(10, random_state=42) #take out 10 patients

    actual = y_test.loc[sample.index] # find the actual diagnoses for the 10 patients
    predicted = best["Logistic Regression"].predict(sample) # use the best parameters for the logistic regression model to predict whether the 10 patients have gallstones or not


    #Show the results in a table with the actual and predicted diagnoses for the 10 patients. The actual diagnoses are mapped to "Gallstones" and "None" for better readability, and the predicted diagnoses are also mapped to "Gallstones" and "None". The results are printed to the console for review.
    results = pd.DataFrame()

    results["Actual"] = actual
    results["Predicted"] = predicted

    results = results.replace({0: "Gallstones", 1: "None"})

    print(results)

    # 9. Save the trained models for explain.py
    joblib.dump((best, X_test, y_test), "models.joblib")


if __name__ == "__main__":
    main()
