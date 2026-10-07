import pandas as pd, numpy as np, warnings, pickle
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu
res,best,Xtr,Xte,ytr,yte,X,y=pickle.load(open('state.pkl','rb'))
X=X.copy(); X['Obesity%']=X['Obesity%'].clip(upper=125.6)   # 1954 is an obvious entry error
cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=2,random_state=7)
mk={'RandForest':lambda:RandomForestClassifier(n_estimators=300,random_state=0,n_jobs=-1),
    'LogReg':lambda:Pipeline([('s',StandardScaler()),('m',LogisticRegression(C=.1,max_iter=5000))])}
pi={k:[] for k in mk}; auc={k:[] for k in mk}
for tr,te in cv.split(X,y):
    for k,f in mk.items():
        m=f().fit(X.iloc[tr],y.iloc[tr]); auc[k].append(roc_auc_score(y.iloc[te],m.predict_proba(X.iloc[te])[:,1]))
        r=permutation_importance(m,X.iloc[te],y.iloc[te],scoring='roc_auc',n_repeats=5,random_state=0)
        pi[k].append(r.importances_mean)
print({k:(round(np.mean(v),3),round(np.std(v),3)) for k,v in auc.items()})
imp=pd.DataFrame({k:np.mean(v,axis=0) for k,v in pi.items()},index=X.columns)
imp['RF_sd']=np.std(pi['RandForest'],axis=0)
imp.to_csv('importance.csv'); print(imp.sort_values('RandForest',ascending=False).head(12).round(4))
print(imp.sort_values('LogReg',ascending=False).head(8).round(4))
# univariate
rows=[]
for c in X.columns:
    a,b=X.loc[y==1,c],X.loc[y==0,c]
    rows.append((c,a.median(),b.median(),mannwhitneyu(a,b).pvalue,roc_auc_score(y,X[c])))
uni=pd.DataFrame(rows,columns=['feat','med_gallstone','med_none','p','auc_uni']).set_index('feat')
uni['dir']=np.where(uni.auc_uni>.5,'higher in gallstone','lower in gallstone')
uni['strength']=(uni.auc_uni-.5).abs()+.5
uni.sort_values('p').to_csv('univariate.csv'); print(uni.sort_values('p').head(12).round(4))
# final models on all data
rf=mk['RandForest']().fit(X,y); lr=mk['LogReg']().fit(X,y)
coef=pd.Series(lr[-1].coef_[0],index=X.columns).sort_values(); coef.to_csv('lr_coef.csv')
# figs
top=imp.sort_values('RandForest',ascending=False).head(12)
fig,ax=plt.subplots(1,2,figsize=(12,5))
t=top.iloc[::-1]; ax[0].barh(t.index,t.RandForest,xerr=t.RF_sd,color='#3a6ea5'); ax[0].set_title('Random forest\npermutation importance (drop in AUC, held-out folds)')
t2=imp.sort_values('LogReg',ascending=False).head(12).iloc[::-1]; ax[1].barh(t2.index,t2.LogReg,color='#c0663a'); ax[1].set_title('Logistic regression\npermutation importance')
plt.tight_layout(); plt.savefig('fig_importance.png',dpi=150); plt.close()
f5=list(top.index[:6]); fig,ax=plt.subplots(figsize=(11,6))
PartialDependenceDisplay.from_estimator(rf,X,f5,ax=ax,n_cols=3,grid_resolution=30); plt.suptitle('Partial dependence of P(gallstone) — random forest'); plt.tight_layout(); plt.savefig('fig_pdp.png',dpi=140); plt.close()
c=coef.iloc[list(range(8))+list(range(-8,0))]; fig,ax=plt.subplots(figsize=(6,5)); ax.barh(c.index,c.values,color=['#3a6ea5' if v<0 else '#c0663a' for v in c.values]); ax.set_title('Logistic regression coefficients (standardized)\nright = ↑ gallstone odds'); plt.tight_layout(); plt.savefig('fig_coef.png',dpi=150); plt.close()
corr=X.corr(method='spearman'); print(corr.loc[top.index[:8],top.index[:8]].round(2))
