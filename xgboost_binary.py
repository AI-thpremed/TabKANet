from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
np.random.seed(0)
from sklearn.metrics import f1_score
import scipy
import os
from pathlib import Path
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--fold', default="1",type=str)
parser.add_argument('--dataset', default="onlineshoper",type=str, help="choose from  [ onlineshoper]")
args = parser.parse_args()
fold=args.fold
print(fold)


if args.dataset=="onlineshoper":
    target_name = 'Revenue'
    task = 'classification'
    continuous_features = ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues']
    categorical_features = ['Administrative', 'Informational', 'ProductRelated', 'SpecialDay', 'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend' ]
    key="onlineshoper"


elif args.dataset=="bankmarketing":
    target_name = 'Output'
    task = 'classification'
    continuous_features = ['age',  'balance',  'duration', 'campaign', 'pdays', 'previous']
    categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact','day', 'month', 'poutcome']
    key="bankmarketing"




train_data = pd.read_csv('/path/templates/'+key+'/Fold'+fold+'/train.csv').fillna('0')

test_data = pd.read_csv('/path/templates/'+key+'/Fold'+fold+'/test.csv').fillna('0')

val_data = pd.read_csv('/path/templates/'+key+'/Fold'+fold+'/val.csv').fillna('0')


label_encoders = {}

all_data = pd.concat([train_data, test_data, val_data])

for feature in categorical_features:
    le = LabelEncoder()
    all_data[feature] = le.fit_transform(all_data[feature])
    label_encoders[feature] = le  

train_data[categorical_features] = all_data[categorical_features].iloc[:len(train_data)]
test_data[categorical_features] = all_data[categorical_features].iloc[len(train_data):len(train_data) + len(test_data)]
val_data[categorical_features] = all_data[categorical_features].iloc[-len(val_data):]

y_train = train_data[target_name]
y_test = test_data[target_name]
y_valid = val_data[target_name]

X_train = train_data[continuous_features + categorical_features]
X_test = test_data[continuous_features + categorical_features]
X_valid = val_data[continuous_features + categorical_features]




clf_xgb = XGBClassifier(max_depth=8,
    learning_rate=0.1,
    n_estimators=1000,
    verbosity=0,
    silent=None,
    objective='binary:logistic',
    booster='gbtree',
    n_jobs=-1,
    nthread=None,
    gamma=0,
    min_child_weight=1,
    max_delta_step=0,
    subsample=0.7,
    colsample_bytree=1,
    colsample_bylevel=1,
    colsample_bynode=1,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    base_score=0.5,
    random_state=0,
    seed=None,
    early_stopping_rounds = 30,
    enable_categorical=True,
    eval_metric='auc'
    )
 
clf_xgb.fit(X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=10)


preds_valid = np.array(clf_xgb.predict_proba(X_valid))
valid_auc = roc_auc_score(y_score=preds_valid[:,1], y_true=y_valid)

preds = np.array(clf_xgb.predict_proba(X_test))
test_auc = roc_auc_score(y_score=preds[:,1], y_true=y_test)

print(f"VALID AUC SCORE FOR {key} : {valid_auc}")
print(f"TEST AUC SCORE FOR {key} : {test_auc}")


valid_f1 = f1_score(y_valid, (preds_valid[:,1] > 0.5).astype(int), average='macro')

test_f1 = f1_score(y_test, (preds[:,1] > 0.5).astype(int), average='macro')

print(f"VALID MACRO F1 SCORE FOR {key} : {valid_f1}")
print(f"TEST MACRO F1 SCORE FOR {key} : {test_f1}")


