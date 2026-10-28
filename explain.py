import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.tree import export_text

from load_data import short


def main(): #Starts the function that holds the code of the script. Nothing runs unit "main" is called at the end.
    models, X_test, y_test = joblib.load("models.joblib")# reads the "models.joblib"file from disk and
                                                         # unpacks what is stored in it into three variables
    print("Models were tuned and fitted on training patients only.")

    short_names = [] #Creates an empty list to fill up
    for column in X_test.columns:#Goes through the column names of X_test, one at a time, each column is one measurement, "column" holds the full name of one measurement per round of the loop.
        short_names.append(short(column)) #Sends full name to the short function, (comes from load data) and turns columns into short labels. The short label is then added to the end of the "short manes" list.
                                          #When the loop is done, short_names has one short label per measurement, in the same order as the columns in X_test.

    importance_scores = {} #Creates an empty dictionary

    for name, model in models.items(): #Goes through the "models" dictionary, one at a time."name" is the model’s name, and "model" is the trained model itself.
        result = permutation_importance(
        model, X_test, y_test, scoring="roc_auc",
        n_repeats=30, random_state=42)#Shuffles one measurement at a time among the test patients, checks how much the model gets worse. Judged in AUC (Area Under Curve), is shuffled 30 times, and same shuffling every time.
        importance_scores[name] = result.importances_mean #Takes the average AUC drop over the 30 shuffles for each measureme and stores it in the dictionary under the model’s name

    importance = pd.DataFrame(importance_scores, index=short_names) #creates a pandas DataFrame from the importance_scores dictionary

    importance["mean_rank"] = importance.rank(ascending=False).mean(axis=1) #ranks the measurements within each model column,ranked for highest AUC to lower ,averages those ranks across each row, goes across models, instead downwards
    importance = importance.sort_values("mean_rank") #Sorts the rows by the mean_rank column, smallest first, and saves the sorted table back into importance, best ranked to the top.
    print("\nTop 10 measurements (drop in AUC per model):")
    print(importance.head(10).round(3))
    importance.round(3).to_csv("importance.csv")#rounds all numbers to 3 decimals and saves the full table to a file called importance.csv

    top10 = importance.head(10).drop(columns="mean_rank").iloc[::-1]
    top10.plot.barh(figsize=(8, 6)) #takes the 10 first (best) rows, removes the mean_rank column from this copy, reverses the row order, draws the horizontal bar chart.
    plt.xlabel("Drop in AUC when the measurement is shuffled") #writes the label under the x-axis
    plt.tight_layout() #adjusts the spacing so the labels and text fit inside the figure
    plt.savefig("fig_4_importance.png", dpi=150) #saves the figure as the image file fig_4_importance.png. dpi=150, and sets the resolution

    print("\nPositive coefficient: towards no gallstones")
    print("Negative coefficient: towards gallstones")
    logistic = models["Logistic Regression"] #picks the logistic regression model out of the models dictionary by its name and stores it in a variable called logistic

    kept = X_test.columns[logistic.named_steps["select"].get_support()] #chooses which measurements are passed on to the classifier, returns a list of True/False values one per measurement in X_test, uses that True/False list as a filter on the column names

    kept_short_names = []
    for column in kept:
        kept_short_names.append(short(column)) #goes through the kept column names, sends each one to short, and adds the short label to kept_short_names

    classifier = logistic.named_steps["model"] #takes the classifier step out of the pipeline. This is the part that holds the coefficients.
    coefficient_values = classifier.coef_[0] #takes the list of coefficients out of the classifier. coef_[0]
    coefficients = pd.Series(coefficient_values, index=kept_short_names) #combines the numbers with the short names
    sorted_coefficients = coefficients.sort_values(key=abs, ascending=False) #takes the abs value, and sorts descendingly
    top_coefficients = sorted_coefficients.head(10)
    print(top_coefficients.round(2))


if __name__ == "__main__": # runs everything,
    main()
