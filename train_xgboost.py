from xgboost import XGBClassifier
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
np.random.seed(0)
from sklearn.metrics import f1_score

import scipy

import os
import wget
from pathlib import Path

from matplotlib import pyplot as plt

 
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--fold', default="1",type=str)
parser.add_argument('--dataset', default="seismic",type=str, help="choose from  [credit onlineshoper biodeg  seismic income  blastchar]")

args = parser.parse_args()
fold=args.fold


print(fold)





if args.dataset=="credit":
    target_name = 'Output'
    task = 'classification'
    continuous_features = ['tab2','tab5','tab8','tab11','tab13']
    categorical_features = [ 'tab1','tab3','tab4','tab6','tab7','tab9','tab10','tab12','tab14','tab15','tab16','tab17','tab18','tab19','tab20']
    key="credit-g"


elif args.dataset=="biodeg":

    target_name = 'Output'
    task = 'classification'
    continuous_features = ['tab1', 'tab2', 'tab8', 'tab12', 'tab13', 'tab14', 'tab15', 'tab17', 'tab18', 'tab22', 'tab27', 'tab28', 'tab30', 'tab31', 'tab36', 'tab37', 'tab39']
    categorical_features = [  'tab38', 'tab32', 'tab33', 'tab34', 'tab35', 'tab29', 'tab3', 'tab4', 'tab5', 'tab6', 'tab7', 'tab9', 'tab10', 'tab11', 'tab16', 'tab19', 'tab20', 'tab21', 'tab23', 'tab24', 'tab25', 'tab26', 'tab40', 'tab41']
    key="biodeg"




elif args.dataset=="onlineshoper":
    target_name = 'Revenue'
    task = 'classification'
    continuous_features = ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues']
    categorical_features = ['Administrative', 'Informational', 'ProductRelated', 'SpecialDay', 'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend' ]
    key="onlineshoper"




train_data = pd.read_csv('/'+key+'/Fold'+fold+'/train.csv').fillna('EMPTY')

test_data = pd.read_csv('/'+key+'/Fold'+fold+'/test.csv').fillna('EMPTY')

val_data = pd.read_csv('/'+key+'/Fold'+fold+'/val.csv').fillna('EMPTY')



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
# print(valid_auc)

preds = np.array(clf_xgb.predict_proba(X_test))
test_auc = roc_auc_score(y_score=preds[:,1], y_true=y_test)
# print(test_auc)

print(f"VALID AUC SCORE FOR {key} : {valid_auc}")
print(f"TEST AUC SCORE FOR {key} : {test_auc}")


valid_f1 = f1_score(y_valid, (preds_valid[:,1] > 0.5).astype(int), average='macro')

test_f1 = f1_score(y_test, (preds[:,1] > 0.5).astype(int), average='macro')

print(f"VALID MACRO F1 SCORE FOR {key} : {valid_f1}")
print(f"TEST MACRO F1 SCORE FOR {key} : {test_f1}")



