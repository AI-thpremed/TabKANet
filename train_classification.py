import sys
import logging
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from models.models import BasicNet,TabKANet ,TabularTransformer
import argparse
CUDA_LAUNCH_BLOCKING=1
from models.metrics import f1_score_macro,auc_score
from models.tools import seed_everything, train, inference, get_dataset, get_data_loader, plot_learning_curve, to_submssion_csv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Hyperparameters for the model
batch_size = 128
inference_batch_size = 128
epochs = 50
early_stopping_patience = 30
seed = 0

parser = argparse.ArgumentParser()

parser.add_argument('--fold', default="1",type=str)
parser.add_argument('--dataset', default="credit",type=str, help="choose from  [credit onlineshoper biodeg  seismic income  blastchar]")
parser.add_argument('--modelname',default="BasicNet", type=str, help="choose from  [BasicNet  TabKANet  TabularTransformer]")

args = parser.parse_args()


fold=args.fold

#MLP
if args.modelname=="BasicNet":
    model_object =  BasicNet
#TabKANet
elif args.modelname=="TabKANet":
    model_object =  TabKANet 

#TabTransformer
elif args.modelname=="TabularTransformer":
    model_object =  TabularTransformer 
    

print(fold)

print(args.modelname)

print(args.dataset)






output_dim = 2
embedding_dim = 64
nhead = 8
num_layers = 3
dim_feedforward = 128
mlp_hidden_dims = [32]
activation = 'relu'
attn_dropout_rate = 0.1
ffn_dropout_rate = 0.1
custom_metric = f1_score_macro
maximize = False
criterion = torch.nn.CrossEntropyLoss()


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




seed_everything(seed)

def train_model():



    train_data = pd.read_csv('/localpath/'+key+'/Fold'+fold+'/train.csv').fillna('EMPTY')

    test_data = pd.read_csv('/localpath/'+key+'/Fold'+fold+'/test.csv').fillna('EMPTY')

    val_data = pd.read_csv('/localpath/'+key+'/Fold'+fold+'/val.csv').fillna('EMPTY')








    train_dataset, test_dataset, val_dataset = \
        get_dataset(
        train_data, test_data, val_data, target_name, 
        task, categorical_features, continuous_features)
    
    train_loader, test_loader, val_loader = \
        get_data_loader(
        train_dataset, test_dataset, val_dataset, 
        train_batch_size=batch_size, inference_batch_size=inference_batch_size)

    vocabulary1=train_dataset.get_vocabulary()
    vocabulary2=test_dataset.get_vocabulary()
    vocabulary3=val_dataset.get_vocabulary()

    
    combined_vocabulary = {}

    for column, mapping in vocabulary1.items():
        if column not in combined_vocabulary:
            combined_vocabulary[column] = mapping
        else:
            combined_vocabulary[column].update(mapping)

    for column, mapping in vocabulary2.items():
        if column not in combined_vocabulary:
            combined_vocabulary[column] = mapping
        else:
            combined_vocabulary[column].update(mapping)

    for column, mapping in vocabulary3.items():
        if column not in combined_vocabulary:
            combined_vocabulary[column] = mapping
        else:
            combined_vocabulary[column].update(mapping)

    final_vocabulary = {}
    for column in combined_vocabulary:
        unique_values = sorted(combined_vocabulary[column].keys())
        final_vocabulary[column] = {value: i for i, value in enumerate(unique_values)}


    model = model_object(
        output_dim=output_dim, 
        vocabulary=final_vocabulary,
        num_continuous_features=len(continuous_features), 
        embedding_dim=embedding_dim, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward, attn_dropout_rate=attn_dropout_rate,
        mlp_hidden_dims=mlp_hidden_dims, activation=activation, ffn_dropout_rate=ffn_dropout_rate
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' if maximize else 'min', factor=0.1, patience=20)
    
    train_history, val_history,test_history = train(
        model, epochs, task, train_loader, val_loader,test_loader ,optimizer, criterion, 
        scheduler=scheduler, custom_metric=custom_metric, 
        maximize=maximize, scheduler_custom_metric=maximize, 
        early_stopping_patience=early_stopping_patience)
    

if __name__ == '__main__':
    train_model()