import sys
import logging
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from tabkanet.models import BasicNet ,BasicNetKAN,TabularTransformer,TabKANet
import argparse
CUDA_LAUNCH_BLOCKING=1
from tabkanet.metrics import f1_score_macro,auc_score
from tabkanet.tools import seed_everything, train, inference, get_dataset, get_data_loader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser()

parser.add_argument('--fold', default="1",type=str)
parser.add_argument('--dataset', default="bankmarketing",type=str, help="choose from  [ bankmarketing    onlineshoper]")
parser.add_argument('--modelname',default="tabtransformer", type=str, help="choose from  [BasicNet tabtransformer kan tabkanet")
parser.add_argument('--gpunum',default=0, type=int)
parser.add_argument('--dim',default=64, type=int)
parser.add_argument('--batch',default=128, type=int)

args = parser.parse_args()

fold=args.fold
batch_size = args.batch
inference_batch_size = 128
epochs = 30
early_stopping_patience = 50
seed = 0




if args.modelname=="BasicNet":
    model_object =  BasicNet 
elif args.modelname=="tabtransformer":
    model_object =  TabularTransformer 
elif args.modelname=="kan":
    model_object =  BasicNetKAN 
elif args.modelname=="tabkanet":
    model_object =  TabKANet 


print(fold)
print(args.modelname)
print(args.dataset)
print(args.batch)


output_dim = 2
embedding_dim = args.dim
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



save_model_path='/path/templates/'+key+'/Fold'+fold+'/'+args.modelname+'.pth'

if args.modelname=="tabkanet" :
    all_count=len(continuous_features)+len(categorical_features)
    if all_count<=10:
        mlp_hidden_dims = [32]
    elif 10<all_count<20 :
        mlp_hidden_dims = [256,32]
    else:
        mlp_hidden_dims = [512,32]



seed_everything(seed)



def train_model():



    train_data = pd.read_csv('/path/templates/'+key+'/Fold'+fold+'/train.csv').fillna('EMPTY')

    test_data = pd.read_csv('/path/templates/'+key+'/Fold'+fold+'/test.csv').fillna('EMPTY')

    val_data = pd.read_csv('/path/templates/'+key+'/Fold'+fold+'/val.csv').fillna('EMPTY')



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
        unique_values = sorted(str(value) for value in combined_vocabulary[column].keys())
        final_vocabulary[column] = {value: i for i, value in enumerate(unique_values)}



    model = model_object(
        output_dim=output_dim, 
        vocabulary=final_vocabulary,
        num_continuous_features=len(continuous_features), 
        embedding_dim=embedding_dim, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward, attn_dropout_rate=attn_dropout_rate,
        mlp_hidden_dims=mlp_hidden_dims, activation=activation, ffn_dropout_rate=ffn_dropout_rate
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' if maximize else 'min', factor=0.1, patience=10)
    
    train_history, val_history,test_history = train(
        model, epochs, task, train_loader, val_loader,test_loader ,optimizer, criterion, 
        scheduler=scheduler, custom_metric=custom_metric, 
        maximize=maximize, scheduler_custom_metric=maximize, 
        early_stopping_patience=early_stopping_patience, gpu_num=args.gpunum)
    

if __name__ == '__main__':
    train_model()