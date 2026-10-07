# Step 2: find the features that are the most important (run analysis.py first)

# Adding necessary libraries
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
np.random.seed(42)


# Load the results of analysis.py
with open('state.pkl', 'rb') as f:
    results, best_models, X_train, X_test, y_train, y_test, X, y = pickle.load(f)

# 1954 is clearly an entry error (the next largest value is 125.6)
X = X.copy()
X['Obesity%'] = X['Obesity%'].clip(upper=125.6)


# The 2 models that we explain
models = {
    'RandForest': lambda: RandomForestClassifier(n_estimators=300, random_state=0),
    'LogReg': lambda: Pipeline([('scaler', StandardScaler()), ('m', LogisticRegression(C=.1, max_iter=5000))]),
}

# Cross-validation: train on the training folds, measure on the validation fold
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=7)

importances = {name: [] for name in models}
aucs = {name: [] for name in models}

for train_idx, val_idx in cv.split(X, y):
    for name, make_model in models.items():
        model = make_model().fit(X.iloc[train_idx], y.iloc[train_idx])

        # AUC on the validation fold
        proba = model.predict_proba(X.iloc[val_idx])[:, 1]
        aucs[name].append(roc_auc_score(y.iloc[val_idx], proba))

        # Importance = drop of the AUC when one feature is shuffled
        r = permutation_importance(model, X.iloc[val_idx], y.iloc[val_idx],
                                   scoring='roc_auc', n_repeats=5, random_state=0)
        importances[name].append(r.importances_mean)

for name in models:
    print(f"{name}: AUC = {np.mean(aucs[name]):.3f} (+/- {np.std(aucs[name]):.3f})")


# Average importance over all folds
importance = pd.DataFrame({name: np.mean(v, axis=0) for name, v in importances.items()}, index=X.columns)
importance['RF_sd'] = np.std(importances['RandForest'], axis=0)
importance.to_csv('importance.csv')

print(importance.sort_values('RandForest', ascending=False).head(12).round(4))
print(importance.sort_values('LogReg', ascending=False).head(8).round(4))


# Compare patients with and without gallstones, one feature at a time
rows = []
for feature in X.columns:
    with_stones = X.loc[y == 1, feature]
    without_stones = X.loc[y == 0, feature]
    p_value = mannwhitneyu(with_stones, without_stones).pvalue
    auc_single = roc_auc_score(y, X[feature])
    rows.append((feature, with_stones.median(), without_stones.median(), p_value, auc_single))

univariate = pd.DataFrame(rows, columns=['feat', 'med_gallstone', 'med_none', 'p', 'auc_uni']).set_index('feat')
univariate['dir'] = np.where(univariate['auc_uni'] > .5, 'higher in gallstone', 'lower in gallstone')
univariate['strength'] = (univariate['auc_uni'] - .5).abs() + .5
univariate.sort_values('p').to_csv('univariate.csv')

print(univariate.sort_values('p').head(12).round(4))


# Train both models on all patients
rf = models['RandForest']().fit(X, y)
lr = models['LogReg']().fit(X, y)

# Coefficients of the logistic regression (features are standardized, so they can be compared)
coef = pd.Series(lr[-1].coef_[0], index=X.columns).sort_values()
coef.to_csv('lr_coef.csv')


# Plot 1: permutation importance
top_rf = importance.sort_values('RandForest', ascending=False).head(12)
top_lr = importance.sort_values('LogReg', ascending=False).head(12)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

data = top_rf.iloc[::-1]
axes[0].barh(data.index, data['RandForest'], xerr=data['RF_sd'], color='#3a6ea5')
axes[0].set_title('Random forest\npermutation importance (drop in AUC, held-out folds)')

data = top_lr.iloc[::-1]
axes[1].barh(data.index, data['LogReg'], color='#c0663a')
axes[1].set_title('Logistic regression\npermutation importance')

for ax in axes:
    ax.set_xlabel('Drop in AUC')
    ax.grid(True, linestyle=':')

plt.tight_layout()
plt.savefig('fig_importance.png', dpi=150)
plt.close()


# Plot 2: partial dependence of the 6 most important features
top6 = list(top_rf.index[:6])

fig, ax = plt.subplots(figsize=(11, 6))
PartialDependenceDisplay.from_estimator(rf, X, top6, ax=ax, n_cols=3, grid_resolution=30)
plt.suptitle('Partial dependence of P(gallstone) - random forest')
plt.tight_layout()
plt.savefig('fig_pdp.png', dpi=140)
plt.close()


# Plot 3: coefficients of the logistic regression (8 most negative and 8 most positive)
selected = coef.iloc[list(range(8)) + list(range(-8, 0))]
colors = ['#3a6ea5' if v < 0 else '#c0663a' for v in selected.values]

fig, ax = plt.subplots(figsize=(6, 5))
ax.barh(selected.index, selected.values, color=colors)
ax.set_xlabel('Coefficient (right = higher odds of gallstone)')
ax.set_title('Logistic regression coefficients (standardized)')
ax.grid(True, linestyle=':')
plt.tight_layout()
plt.savefig('fig_coef.png', dpi=150)
plt.close()


# Correlation between the most important features
corr = X.corr(method='spearman')
print(corr.loc[top_rf.index[:8], top_rf.index[:8]].round(2))
