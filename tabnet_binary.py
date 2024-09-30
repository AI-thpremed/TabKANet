from pytorch_tabnet.tab_model import TabNetClassifier
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
os.environ['CUDA_VISIBLE_DEVICES'] = f"0,1,2,3,4,5,6,7"
import argparse

from pytorch_tabnet.augmentations import ClassificationSMOTE


parser = argparse.ArgumentParser()
parser.add_argument('--fold', default="1",type=str)
parser.add_argument('--dataset', default="onlineshoper",type=str, help="choose from  [bankmarketing onlineshoper]")
parser.add_argument('--gpunum',default=0, type=int)
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

tabnet_params = {"cat_idxs":cat_idxs,
                 "cat_dims":cat_dims,
                 "cat_emb_dim":64,
                 "optimizer_fn":torch.optim.Adam,
                 "optimizer_params":dict(lr=1e-3),
                 "scheduler_params":{"step_size":50, # how to use learning rate scheduler
                                 "gamma":0.9},
                 "scheduler_fn":torch.optim.lr_scheduler.StepLR,
                 "mask_type":'entmax', # "sparsemax"
                 "grouped_features" : grouped_features
                }
clf = TabNetClassifier(**tabnet_params
                      )

max_epochs = 100 if not os.getenv("CI", False) else 2

aug = ClassificationSMOTE(p=0.2)

# This illustrates the warm_start=False behaviour
save_history = []

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).to(device)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).to(device)

# Fitting the model without starting from a warm start nor computing the feature importance
clf.fit(
    X_train=X_train_tensor.cpu().numpy(),  # TabNet expects numpy arrays
    y_train=y_train_tensor.cpu().numpy(),
    eval_set=[(X_train_tensor.cpu().numpy(), y_train_tensor.cpu().numpy()), 
              (X_valid_tensor.cpu().numpy(), y_valid_tensor.cpu().numpy())],
    eval_name=['train', 'valid'],
    eval_metric=['auc'],
    max_epochs=max_epochs,
    patience=30,
    batch_size=512,
    virtual_batch_size=128,
    num_workers=0,
    weights=1,
    drop_last=False,
    augmentations=aug,  # aug, None
    compute_importance=False
)

save_history.append(clf.history["valid_auc"])

preds_valid = np.array(clf.predict_proba(X_valid))
valid_auc = roc_auc_score(y_score=preds_valid[:,1], y_true=y_valid)

preds = np.array(clf.predict_proba(X_test))
test_auc = roc_auc_score(y_score=preds[:,1], y_true=y_test)

print(f"VALID AUC SCORE FOR {key} : {valid_auc}")
print(f"TEST AUC SCORE FOR {key} : {test_auc}")
