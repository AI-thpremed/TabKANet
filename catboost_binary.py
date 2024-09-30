from catboost import Pool, CatBoostClassifier
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
parser.add_argument('--dataset', default="onlineshoper",type=str, help="choose from  [onlineshoper   bankmarketing]")
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




train_data = pd.read_csv('/path/templates/'+key+'/Fold'+fold+'/train.csv')

test_data = pd.read_csv('/path/templates/'+key+'/Fold'+fold+'/test.csv')

val_data = pd.read_csv('/path/templates/'+key+'/Fold'+fold+'/val.csv')





label_encoders = {}

all_data = pd.concat([train_data, test_data, val_data])

for feature in categorical_features:
    le = LabelEncoder()
    all_data[feature] = le.fit_transform(all_data[feature])
    label_encoders[feature] = le  # 保存每个特征的编码器以便将来使用


train_data[categorical_features] = all_data[categorical_features].iloc[:len(train_data)].astype(int)
test_data[categorical_features] = all_data[categorical_features].iloc[len(train_data):len(train_data) + len(test_data)].astype(int)
val_data[categorical_features] = all_data[categorical_features].iloc[-len(val_data):].astype(int)

train_data[categorical_features] = train_data[categorical_features].astype(int)
test_data[categorical_features] = test_data[categorical_features].astype(int)
val_data[categorical_features] = val_data[categorical_features].astype(int)


y_train = train_data[target_name]
y_test = test_data[target_name]
y_valid = val_data[target_name]


X_train = train_data[categorical_features + continuous_features]
X_test = test_data[categorical_features + continuous_features]
X_valid = val_data[categorical_features + continuous_features]


features = categorical_features + continuous_features
categorical_features_indices = [i for i, f in enumerate(features) if f in categorical_features]




model = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, loss_function='Logloss', eval_metric='AUC', random_seed=99, od_type='Iter', od_wait=100) 
model.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_valid, y_valid),plot=True)


preds_valid = np.array(model.predict_proba(X_valid))
valid_auc = roc_auc_score(y_score=preds_valid[:,1], y_true=y_valid)

preds = np.array(model.predict_proba(X_test))
test_auc = roc_auc_score(y_score=preds[:,1], y_true=y_test)

print(f"VALID AUC SCORE FOR {key} : {valid_auc}")
print(f"TEST AUC SCORE FOR {key} : {test_auc}")


