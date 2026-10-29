# Predicting gallstone disease from clinical measurements

DAT540 group project (group Sekssyv): Simen Helland, Chantel Helleland, Parsa Moniri, Theodor Solland, Martin Lærum.

**Research question:** Can we distinguish patients with gallstones from patients without, using clinical measurements?

We train and compare three machine learning models (logistic regression, decision tree and random forest) on 319 patients, evaluate them on patients they have never seen, and check which measurements they rely on.

## Dataset

`dataset-uci.xlsx` is the [Gallstone dataset](https://archive.ics.uci.edu/dataset/1150/gallstone-1) from the UCI Machine Learning Repository: 319 patients from a clinic in Ankara (2022–2023), with 38 measurements each (demographics, bioimpedance body composition and blood tests).

The target column is `Gallstone Status`. Following the dataset's own coding, **0 = gallstones** (161 patients) and **1 = no gallstones** (158 patients).

## Files

| File | What it does |
|---|---|
| `load_data.py` | Reads the dataset and cleans it: drops missing and duplicate rows, and replaces one impossible Obesity value (1954%) with the column median. Returns the measurements (`X`) and the diagnosis (`y`). Also contains `short()`, which shortens column names for plots. |
| `explore.py` | Explores the data: prints a data overview, flags extreme values, and draws the class balance, boxplots and correlation heatmap. |
| `analysis.py` | Splits the data (75% training, 25% test), tunes each model with grid search and stratified 5-fold cross-validation, compares them with a baseline, prints the results, and saves the trained models to `models.joblib`. |
| `explain.py` | Loads the saved models, measures which measurements matter with permutation importance, and prints the logistic regression coefficients. |
| `requirements.txt` | The Python packages needed. |

## How to run

Put all files in the same folder, then run from that folder:

```bash
pip install -r requirements.txt
python explore.py
python analysis.py
python explain.py
```

On Mac, use `pip3` and `python3`. On Windows, if `python` isn't found, use `py` instead.

`analysis.py` must run before `explain.py`, because `explain.py` loads the models that `analysis.py` saves. All random steps use `random_state=42`, so every run gives the same results.

## Outputs

| File | Made by | Shows |
|---|---|---|
| `fig_1_class_balance.png` | `explore.py` | Number of patients with and without gallstones |
| Boxplots (opens in browser) | `explore.py` | The six measurements most linked to the diagnosis, compared between groups |
| `fig_3_correlation.png` | `explore.py` | How the measurements correlate with each other |
| `models.joblib` | `analysis.py` | The three trained models and the test patients |
| `importance.csv` | `explain.py` | Permutation importance of every measurement, per model |
| `fig_4_importance.png` | `explain.py` | The 10 most important measurements |

## Method

1. **Cleaning:** no missing values or duplicates were found; one data entry error was fixed with median imputation.
2. **Split:** 25% of patients (80) are held out as a test set and only used at the end.
3. **Pipeline:** each model runs in the same pipeline: MinMax scaling to 0–1, a feature selection step (`SelectKBest`, set to keep all features), then the model. Scaling is learned only from training data, which prevents data leakage.
4. **Tuning:** `GridSearchCV` tries every combination of settings with stratified 5-fold cross-validation on the training data and picks the one with the best ROC-AUC.
5. **Evaluation:** each tuned model is tested once on the held-out patients (accuracy, confusion matrix, gallstone recall and ROC-AUC) and compared with a baseline that always guesses the most common class.
6. **Explanation:** permutation importance shuffles one measurement at a time and measures the drop in ROC-AUC.

## Results

| Model | Cross-validation AUC | Test accuracy | Test ROC-AUC | Gallstone recall |
|---|---|---|---|---|
| Logistic regression | 0.841 | 78.7% | 0.897 | 89.7% |
| Random forest | 0.829 | 81.2% | 0.901 | 87.2% |
| Decision tree | 0.755 | 73.8% | 0.779 | 89.7% |
| Baseline (most common class) | | 48.7% | | |

- All three models clearly beat the baseline. Logistic regression and the random forest are essentially tied.
- **Vitamin D and CRP** are the most important measurements across all three models.

## Limitations

- The dataset is small (319 patients) and comes from a single clinic, with one train/test split, so the results need confirming on new data.
- The results show associations, not causes.
- Following the dataset's labels, the direction of several effects (for example vitamin D, CRP and body fat) is the opposite of published research, which suggests the labels may be swapped in the original data. This doesn't affect model performance, only the direction of the findings.
