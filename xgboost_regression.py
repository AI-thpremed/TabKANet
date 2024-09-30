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
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


parser = argparse.ArgumentParser()
parser.add_argument('--fold', default="1",type=str)
parser.add_argument('--dataset', default="cahouse",type=str, help="choose from  [ cahouse ]")
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
 



print(args.dataset)





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


clf_xgb = XGBRegressor(max_depth=8,
                        learning_rate=0.1,
                        n_estimators=1000,
                        verbosity=0,
                        objective='reg:squarederror',  
                        booster='gbtree',
                        n_jobs=-1,
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
                        early_stopping_rounds=30,
                        eval_metric='rmse',  
                        enable_categorical=True
                        )


clf_xgb.fit(X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=10)


preds_valid = clf_xgb.predict(X_valid)
preds = clf_xgb.predict(X_test)


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
