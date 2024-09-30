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
import os
os.environ['CUDA_VISIBLE_DEVICES'] = f"0,1,2,3,4,5,6,7"
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, cohen_kappa_score, recall_score
import argparse

from pytorch_tabnet.augmentations import ClassificationSMOTE

parser = argparse.ArgumentParser()
parser.add_argument('--fold', default="1",type=str)
parser.add_argument('--dataset', default="multi_seg",type=str, help="choose from  [multi_forest    multi_seg]")
parser.add_argument('--gpunum',default=0, type=int)
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



tabnet_params = {"output_dim":num_classes,
    "cat_idxs":cat_idxs,
                 "cat_dims":cat_dims,
                 "cat_emb_dim":64,
                 "optimizer_fn":torch.optim.Adam,
                 "optimizer_params":dict(lr=2e-2),
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
    eval_metric=['logloss'],
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



preds_valid = np.array(clf.predict_proba(X_valid))
preds = np.array(clf.predict_proba(X_test))

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
