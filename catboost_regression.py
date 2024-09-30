from catboost import Pool, CatBoostClassifier

import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
np.random.seed(0)
from sklearn.metrics import f1_score
import scipy
import os
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

parser = argparse.ArgumentParser()
parser.add_argument('--fold', default="1",type=str)
parser.add_argument('--dataset', default="cahouse",type=str, help="choose from  [ cahouse  cpu_small sarcos]")

args = parser.parse_args()
fold=args.fold
print(fold)



if args.dataset=="cahouse":
    target_name = 'Output'
    task = 'regression'
    continuous_features = [  'longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
    categorical_features = ['ocean_proximity']
    key = "cahouse"

elif args.dataset=="sarcos":
    target_name = 'V22'
    task = 'regression'
    continuous_features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']
    categorical_features = []
    key = "sarcos"

elif args.dataset=="cpu_small":
    target_name = 'usr'
    task = 'regression'
    continuous_features = ['lread', 'lwrite', 'scall', 'sread', 'swrite', 'fork', 'exec', 'rchar', 'wchar', 'runqsz', 'freemem', 'freeswap']
    categorical_features = []
    key = "cpu_small"
 


train_data = pd.read_csv('/path/templates/'+key+'/Fold'+fold+'/train.csv')
test_data = pd.read_csv('/path/templates/'+key+'/Fold'+fold+'/test.csv')
val_data = pd.read_csv('/path/templates/'+key+'/Fold'+fold+'/val.csv')


label_encoders = {}
all_data = pd.concat([train_data, test_data, val_data])

for feature in categorical_features:
    le = LabelEncoder()
    all_data[feature] = le.fit_transform(all_data[feature])
    label_encoders[feature] = le  

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

model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, 
                          loss_function='RMSE', random_seed=99, od_type='Iter', od_wait=100)

model.fit(X_train, y_train, cat_features=categorical_features_indices, eval_set=(X_valid, y_valid), plot=True)

preds_valid = np.array(model.predict(X_valid))
preds = np.array(model.predict(X_test))

def compute_regression_metrics(predictions, targets):
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mse)
    corr = r2_score(targets, predictions)
    return mse, mae, rmse, corr

valid_mse, valid_mae, valid_rmse, valid_corr = compute_regression_metrics(y_valid, preds_valid)
test_mse, test_mae, test_rmse, test_corr = compute_regression_metrics(y_test, preds)

print(f"VALID MSE: {valid_mse}, MAE: {valid_mae}, RMSE: {valid_rmse}, CORR: {valid_corr}")
print(f"TEST MSE: {test_mse}, MAE: {test_mae}, RMSE: {test_rmse}, CORR: {test_corr}")




