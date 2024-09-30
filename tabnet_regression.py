from pytorch_tabnet.tab_model import TabNetRegressor
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
np.random.seed(0)
import scipy
from sklearn.metrics import f1_score
import os
from pathlib import Path
from matplotlib import pyplot as plt

import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
os.environ['CUDA_VISIBLE_DEVICES'] = f"0,1,2,3,4,5,6,7"
import argparse



parser = argparse.ArgumentParser()

parser.add_argument('--fold', default="1",type=str)
parser.add_argument('--dataset', default="cahouse",type=str, help="choose from  [cahouse sarcos ]")
parser.add_argument('--gpunum',default=1, type=int)

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

y_train = train_data[target_name].values
y_test = test_data[target_name].values
y_valid = val_data[target_name].values

X_train = train_data[categorical_features+continuous_features ].values
X_test = test_data[categorical_features+continuous_features].values
X_valid = val_data[categorical_features+continuous_features].values


features = categorical_features + continuous_features



categorical_dims = {col: all_data[col].nunique() for col in categorical_features}


cat_idxs = [i for i, f in enumerate(features) if f in categorical_features]
cat_dims = [categorical_dims[f] for f in categorical_features]


grouped_features = []


device = torch.device("cuda:"+str(args.gpunum) if torch.cuda.is_available() else "cpu")
tabnet_params = {
    "cat_idxs": cat_idxs,
    "cat_dims": cat_dims,
    "cat_emb_dim": 64,
    "optimizer_fn": torch.optim.Adam,
    "optimizer_params": dict(lr=3e-3),
    "scheduler_params": {"step_size": 50, "gamma": 0.9},
    "scheduler_fn": torch.optim.lr_scheduler.StepLR,
    "mask_type": 'entmax',
    "grouped_features": grouped_features
}

clf = TabNetRegressor(**tabnet_params)

max_epochs = 500 if not os.getenv("CI", False) else 2

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(device) 
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).to(device)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).reshape(-1, 1).to(device)  

# Fitting the model without starting from a warm start nor computing the feature importance
clf.fit(
    X_train=X_train_tensor.cpu().numpy(),
    y_train=y_train_tensor.cpu().numpy(),
    eval_set=[(X_train_tensor.cpu().numpy(), y_train_tensor.cpu().numpy()),
              (X_valid_tensor.cpu().numpy(), y_valid_tensor.cpu().numpy())],
    eval_name=['train', 'valid'],
    max_epochs=max_epochs,
    patience=100,
    batch_size=512,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False,
    compute_importance=False
)


preds_valid = clf.predict(X_valid)
preds = clf.predict(X_test)  # 

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
