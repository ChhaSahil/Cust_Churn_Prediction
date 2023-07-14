import pickle

import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import ADASYN

df = pd.read_csv('Bank Customer Churn Prediction.csv')
df.drop(['customer_id'], axis = 1, inplace = True)
X = df.drop(['churn'], axis=1)
y = df.churn

le = LabelEncoder()
X['gender'] = le.fit_transform(X['gender'])
X['country'] = le.fit_transform(X['country'])

def cat_score(score):
  if 800<=score<=850:
    return 4
  elif 700<=score<800:
    return 3
  elif 600<=score<700:
    return 2
  else:
    return 1

X['cat_score'] = X['credit_score'].apply(cat_score)
X.drop(['credit_score'],axis=1,inplace=True)
X['products_number'] = X['products_number'].astype('float64')
X['age'] = X['age'].astype('float64')
X['tenure'] = X['tenure'].astype('float64')
categorical_features_indices = np.where(X.dtypes != np.float)[0]

X_out, y_out = ADASYN(random_state=42).fit_resample(X,y)

n_splits=10
fold=0
scores_p=[]
scores_r = []
kfold = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 11)
xgbc = XGBClassifier(scale_pos_weight = 1)

for train_idx,val_idx in kfold.split(X_out,y_out):
  df_train = X_out.iloc[train_idx]
  df_val = X_out.iloc[val_idx]

  y_train = y_out.iloc[train_idx]
  y_test = y_out.iloc[val_idx]
  xgbc.fit(df_train,y_train)
  pred_xgbc_f = xgbc.predict(df_val)
  scores_p.append(precision_score(y_test,pred_xgbc_f))
  scores_r.append(recall_score(y_test,pred_xgbc_f))
  fold+=1
print('Precision Score - ',end='')
print(np.mean(scores_p))
print('Recall Score - ',end='')
print(np.mean(scores_r))
output_file = 'model_save.sav'
with open(output_file, 'wb') as f_out:
    pickle.dump(xgbc,f_out)
