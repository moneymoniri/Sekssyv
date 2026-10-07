import pandas as pd, numpy as np, warnings, json
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_validate, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.inspection import permutation_importance
df=pd.read_csv('d.tsv',sep='\t')
short=['Status','Age','Gender','Comorbidity','CAD','Hypothyroidism','Hyperlipidemia','DM','Height','Weight','BMI','TBW','ECW','ICW','ECF/TBW','TBFR','LM','Protein','VFR','BM','MM','Obesity%','TFC','VFA','VMA','HFA','Glucose','TC','LDL','HDL','Triglyceride','AST','ALT','ALP','Creatinine','GFR','CRP','HGB','VitaminD']
df.columns=short
y=(df.iloc[:,0]==0).astype(int)   # 1 = gallstone present (UCI codes gallstone as 0)
X=df.iloc[:,1:]
print('pos',y.sum(),'neg',(1-y).sum())
ob='Obesity%'
print(X[ob].sort_values().tail(5).values)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=42)
models={
 'LogReg':(Pipeline([('s',StandardScaler()),('m',LogisticRegression(max_iter=5000))]),{'m__C':[.01,.1,1,10]}),
 'SVM-RBF':(Pipeline([('s',StandardScaler()),('m',SVC(probability=True,random_state=0))]),{'m__C':[.1,1,10],'m__gamma':['scale',.01]}),
 'RandForest':(RandomForestClassifier(n_estimators=300,random_state=0,n_jobs=-1),{'max_depth':[4,None],'min_samples_leaf':[1,3]}),
 'GradBoost':(GradientBoostingClassifier(random_state=0),{'n_estimators':[150],'max_depth':[2,3],'learning_rate':[.05],'subsample':[.8]}),
}
inner=StratifiedKFold(5,shuffle=True,random_state=1)
outer=RepeatedStratifiedKFold(n_splits=5,n_repeats=1,random_state=2)
res={};best={}
for n,(est,g) in models.items():
    gs=GridSearchCV(est,g,cv=inner,scoring='roc_auc',n_jobs=-1)
    cv=cross_validate(gs,Xtr,ytr,cv=outer,scoring=['roc_auc','accuracy','f1','recall','precision'],n_jobs=1)
    gs.fit(Xtr,ytr); best[n]=gs
    p=gs.predict_proba(Xte)[:,1]; pr=(p>.5).astype(int)
    res[n]=dict(cv_auc=cv['test_roc_auc'].mean(),cv_auc_sd=cv['test_roc_auc'].std(),cv_acc=cv['test_accuracy'].mean(),
      test_auc=roc_auc_score(yte,p),test_acc=accuracy_score(yte,pr),test_f1=f1_score(yte,pr),test_sens=recall_score(yte,pr),
      test_spec=recall_score(yte,pr,pos_label=0),params=gs.best_params_)
    print(n,{k:(round(v,3) if not isinstance(v,dict) else v) for k,v in res[n].items()},flush=True)
import pickle;pickle.dump((res,best,Xtr,Xte,ytr,yte,X,y),open('state.pkl','wb'))
