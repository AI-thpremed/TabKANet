import sys
import logging
import torch
import pandas as pd
from tabkanet.models import BasicNet ,BasicNetKAN,TabularTransformer,TabKANet
import argparse
CUDA_LAUNCH_BLOCKING=1
from tabkanet.metrics import f1_score_macro,auc_score
from tabkanet.tools_multiclass import seed_everything, train, get_dataset, get_data_loader
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

batch_size = 128
inference_batch_size = 128
epochs = 120
early_stopping_patience = 120
seed = 0

parser = argparse.ArgumentParser()
parser.add_argument('--fold', default="1",type=str)
parser.add_argument('--dataset', default="multi_seg",type=str, help="choose from  [multi_forest    multi_seg ]")
parser.add_argument('--modelname',default="tabkanet", type=str, help="choose from  [BasicNet tabtransformer kan tabkanet")
parser.add_argument('--gpunum',default=4, type=int)
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
print(fold)
print(args.gpunum)


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

output_dim = num_classes

# save_model_path='/templates/'+key+'/Fold'+fold+'/'+args.modelname+'.pth'


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



    train_data = pd.read_csv('/path/templates/'+key+'/Fold'+fold+'/train.csv')

    test_data = pd.read_csv('/path/templates/'+key+'/Fold'+fold+'/test.csv')

    val_data = pd.read_csv('/path/templates/'+key+'/Fold'+fold+'/val.csv')








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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' if maximize else 'min', factor=0.1, patience=20)
    


    train_history, val_history,test_history = train(
        model, epochs, task, train_loader, val_loader,test_loader ,optimizer, criterion, 
        scheduler=scheduler, custom_metric=custom_metric, 
        maximize=maximize, scheduler_custom_metric=maximize, 
        early_stopping_patience=early_stopping_patience, gpu_num=args.gpunum)
    

if __name__ == '__main__':
    train_model()