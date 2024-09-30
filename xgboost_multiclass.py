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
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, cohen_kappa_score, recall_score
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--fold', default="1",type=str)
parser.add_argument('--dataset', default="multi_seg",type=str, help="choose from  [multi_forest    multi_seg]")
args = parser.parse_args()


fold=args.fold


print(fold)




if args.dataset=="multi_forest":
    target_name = 'class'
    task = 'classification'
    continuous_features = ['elevation', 'aspect', 'slope', 'horizontal_distance_to_hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']
    categorical_features = [  'wilderness_area1', 'wilderness_area2', 'wilderness_area3', 'wilderness_area4', 'soil_type_1', 'soil_type_2', 'soil_type_3', 'soil_type_4', 'soil_type_5', 'soil_type_6', 'soil_type_7', 'soil_type_8', 'soil_type_9', 'soil_type_10', 'soil_type_11', 'soil_type_12', 'soil_type_13', 'soil_type_14', 'soil_type_15', 'soil_type_16', 'soil_type_17', 'soil_type_18', 'soil_type_19', 'soil_type_20', 'soil_type_21', 'soil_type_22', 'soil_type_23', 'soil_type_24', 'soil_type_25', 'soil_type_26', 'soil_type_27', 'soil_type_28', 'soil_type_29', 'soil_type_30', 'soil_type_31', 'soil_type_32', 'soil_type_33', 'soil_type_34', 'soil_type_35', 'soil_type_36', 'soil_type_37', 'soil_type_38', 'soil_type_39', 'soil_type_40']
    key="multi_forest"
    num_classes = 7


elif args.dataset=="multi_seg":
    target_name = 'class'
    task = 'classification'
    continuous_features = ['region.centroid.col', 'region.centroid.row', 'region.pixel.count', 'short.line.density.5', 'short.line.density.2', 'vedge.mean', 'vegde.sd', 'hedge.mean', 'hedge.sd', 'intensity.mean', 'rawred.mean', 'rawblue.mean', 'rawgreen.mean', 'exred.mean', 'exblue.mean', 'exgreen.mean', 'value.mean', 'saturation.mean', 'hue.mean']
    categorical_features = []
    key="multi_seg"

    num_classes = 7

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
    objective='multi:softprob',
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
    early_stopping_rounds=100,
    enable_categorical=True,
    eval_metric='mlogloss',
    num_class=num_classes
)

clf_xgb.fit(X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=10)

preds_valid = np.array(clf_xgb.predict_proba(X_valid))
preds = np.array(clf_xgb.predict_proba(X_test))

valid_auc = roc_auc_score(y_valid, preds_valid, multi_class='ovr')
test_auc = roc_auc_score(y_test, preds, multi_class='ovr')

valid_f1 = f1_score(y_valid, np.argmax(preds_valid, axis=1), average='macro')
test_f1 = f1_score(y_test, np.argmax(preds, axis=1), average='macro')

valid_macro_acc = accuracy_score(y_valid, np.argmax(preds_valid, axis=1))
test_macro_acc = accuracy_score(y_test, np.argmax(preds, axis=1))

valid_kappa = cohen_kappa_score(y_valid, np.argmax(preds_valid, axis=1))
test_kappa = cohen_kappa_score(y_test, np.argmax(preds, axis=1))

valid_recall = recall_score(y_valid, np.argmax(preds_valid, axis=1), average='macro')
test_recall = recall_score(y_test, np.argmax(preds, axis=1), average='macro')

print(f"VALID AUC SCORE FOR {key} : {valid_auc}")
print(f"TEST AUC SCORE FOR {key} : {test_auc}")
print(f"VALID MACRO F1 SCORE FOR {key} : {valid_f1}")
print(f"TEST MACRO F1 SCORE FOR {key} : {test_f1}")
print(f"VALID MACRO ACCURACY FOR {key} : {valid_macro_acc}")
print(f"TEST MACRO ACCURACY FOR {key} : {test_macro_acc}")
print(f"VALID KAPPA SCORE FOR {key} : {valid_kappa}")
print(f"TEST KAPPA SCORE FOR {key} : {test_kappa}")
print(f"VALID RECALL FOR {key} : {valid_recall}")
print(f"TEST RECALL FOR {key} : {test_recall}")