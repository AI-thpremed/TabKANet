import sys
sys.path.append('../') 
import logging
import torch
from tabkanet.models import BasicNet ,BasicNetKAN,TabularTransformer,TabKANet
import pandas as pd
from tabkanet.metrics import root_mean_squared_logarithmic_error,root_mean_squared_error
from tabkanet.tools_regression import seed_everything, train, get_data, get_dataset, get_data_loader
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')



parser = argparse.ArgumentParser()
parser.add_argument('--fold', default="1",type=str)
parser.add_argument('--dataset', default="cpu_small",type=str, help="choose from  [baseball cahouse  cpu_small]")
parser.add_argument('--modelname',default="BasicNet", type=str, help="choose from  [BasicNet tabtransformer kan tabkanet")
parser.add_argument('--gpunum',default=0, type=int)

args = parser.parse_args()

fold=args.fold



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

batch_size = 64
inference_batch_size = 128
epochs = 300
early_stopping_patience = 100
early_stopping_start_from = 200
seed = 0
output_dim = 1
embedding_dim = 32
nhead = 8
num_layers = 3
dim_feedforward = 8
mlp_hidden_dims = [32]
activation = 'relu'
custom_metric = root_mean_squared_error
# RMSE
maximize = False
# Loss function for the model
criterion = torch.nn.MSELoss()
attn_dropout_rate = 0.1
ffn_dropout_rate = 0.1



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




if args.modelname=="tabkanet" :
    all_count=len(continuous_features)+len(categorical_features)
    if all_count<=10:
        mlp_hidden_dims = [32]
    elif 10<all_count<20 :
        mlp_hidden_dims = [256,32]
    else:
        mlp_hidden_dims = [512,32]








def train_model():
    train_data = pd.read_csv('/path/templates/'+key+'/Fold'+fold+'/train.csv').fillna('0')
    test_data = pd.read_csv('/path/templates/'+key+'/Fold'+fold+'/test.csv').fillna('0')
    val_data = pd.read_csv('/path/templates/'+key+'/Fold'+fold+'/val.csv').fillna('0')




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

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' if maximize else 'min', factor=0.1, patience=10)

    
    train_history, val_history,test_history = train(
        model, epochs, task, train_loader, val_loader, test_loader,optimizer, criterion, 
        scheduler=scheduler, custom_metric=custom_metric, 
        maximize=maximize, scheduler_custom_metric=maximize, 
        early_stopping_patience=early_stopping_patience, early_stopping_start_from=early_stopping_start_from, gpu_num=args.gpunum)
    

if __name__ == '__main__':
    train_model()