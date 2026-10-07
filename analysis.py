# Step 1: train and compare 4 models (run this file first)

# Adding necessary libraries
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score
from load_data import load_xlsx

warnings.filterwarnings('ignore')
np.random.seed(42)


# Read the dataset
path = 'dataset-uci.xlsx'
df = load_xlsx(path)
print(df.shape)

# Short names for the 39 columns (same order as in the Excel file)
short_names = ['Status', 'Age', 'Gender', 'Comorbidity', 'CAD', 'Hypothyroidism', 'Hyperlipidemia', 'DM',
               'Height', 'Weight', 'BMI', 'TBW', 'ECW', 'ICW', 'ECF/TBW', 'TBFR', 'LM', 'Protein', 'VFR', 'BM',
               'MM', 'Obesity%', 'TFC', 'VFA', 'VMA', 'HFA', 'Glucose', 'TC', 'LDL', 'HDL', 'Triglyceride',
               'AST', 'ALT', 'ALP', 'Creatinine', 'GFR', 'CRP', 'HGB', 'VitaminD']
df.columns = short_names


# Target y: 1 = gallstones present (UCI codes gallstones as 0)
y = (df['Status'] == 0).astype(int)

# Features X: all other columns
X = df.drop(columns='Status')

print(f"Gallstones: {y.sum()} | No gallstones: {(1 - y).sum()}")

# The largest Obesity% values (one value is clearly wrong)
print(X['Obesity%'].sort_values().tail(5).values)


# Split in training (75%) and test (25%), same class ratio in both
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, stratify=y, random_state=42)
print(f"Training: {X_train.shape} | Test: {X_test.shape}")


# The 4 models and the settings that are tried for each of them
# Note: 'm__C' means the parameter C of the step called 'm' in the pipeline
models = {
    'LogReg': (
        Pipeline([('scaler', StandardScaler()), ('m', LogisticRegression(max_iter=5000))]),
        {'m__C': [.01, .1, 1, 10]}
    ),
    'SVM-RBF': (
        Pipeline([('scaler', StandardScaler()), ('m', SVC(probability=True, random_state=0))]),
        {'m__C': [.1, 1, 10], 'm__gamma': ['scale', .01]}
    ),
    'RandForest': (
        RandomForestClassifier(n_estimators=300, random_state=0),
        {'max_depth': [4, None], 'min_samples_leaf': [1, 3]}
    ),
    'GradBoost': (
        GradientBoostingClassifier(random_state=0),
        {'n_estimators': [150], 'max_depth': [2, 3], 'learning_rate': [.05], 'subsample': [.8]}
    ),
}

# Inner loop: choose the best settings / outer loop: honest estimate of the performance
inner = StratifiedKFold(5, shuffle=True, random_state=1)
outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=2)


# Train and evaluate every model
results = {}
best_models = {}

for name, (model, settings) in models.items():

    # Search for the best settings (score = AUC)
    search = GridSearchCV(model, settings, cv=inner, scoring='roc_auc')

    # Cross-validation on the training set
    cv = cross_validate(search, X_train, y_train, cv=outer, scoring=['roc_auc', 'accuracy', 'f1', 'recall', 'precision'])

    # Train the final model on the whole training set
    search.fit(X_train, y_train)
    best_models[name] = search

    # Predict on the test set
    proba = search.predict_proba(X_test)[:, 1]    # probability of gallstones
    pred = (proba > .5).astype(int)               # yes / no

    results[name] = dict(
        cv_auc=cv['test_roc_auc'].mean(),
        cv_auc_sd=cv['test_roc_auc'].std(),
        cv_acc=cv['test_accuracy'].mean(),
        test_auc=roc_auc_score(y_test, proba),
        test_acc=accuracy_score(y_test, pred),
        test_f1=f1_score(y_test, pred),
        test_sens=recall_score(y_test, pred),                # gallstone patients found
        test_spec=recall_score(y_test, pred, pos_label=0),   # healthy people found
        params=search.best_params_
    )

    r = results[name]
    print(f"{name}: CV AUC = {r['cv_auc']:.3f} (+/- {r['cv_auc_sd']:.3f}) | Test AUC = {r['test_auc']:.3f} | Test accuracy = {r['test_acc']:.3f}")
    print(f"    sensitivity = {r['test_sens']:.3f} | specificity = {r['test_spec']:.3f} | settings = {r['params']}")


# Save everything for explain.py
with open('state.pkl', 'wb') as f:
    pickle.dump((results, best_models, X_train, X_test, y_train, y_test, X, y), f)
