nohup python train_classification.py  --dataset  bankmarketing --modelname tabkanet --fold 1 >tabkanet_bankmarketing_fold1.log&
nohup python train_classification.py  --dataset  bankmarketing --modelname tabkanet --fold 2 >tabkanet_bankmarketing_fold2.log&
nohup python train_classification.py  --dataset  bankmarketing --modelname tabkanet --fold 3 >tabkanet_bankmarketing_fold3.log&
nohup python train_classification.py  --dataset  bankmarketing --modelname tabkanet --fold 4 >tabkanet_bankmarketing_fold4.log&
nohup python train_classification.py  --dataset  bankmarketing --modelname tabkanet --fold 5 >tabkanet_bankmarketing_fold5.log&

nohup python train_classification.py  --dataset  bankmarketing --modelname tabtransformer --fold 1 >tabtransformer_bankmarketing_fold1.log&
nohup python train_classification.py  --dataset  bankmarketing --modelname tabtransformer --fold 2 >tabtransformer_bankmarketing_fold2.log&
nohup python train_classification.py  --dataset  bankmarketing --modelname tabtransformer --fold 3 >tabtransformer_bankmarketing_fold3.log&
nohup python train_classification.py  --dataset  bankmarketing --modelname tabtransformer --fold 4 >tabtransformer_bankmarketing_fold4.log&
nohup python train_classification.py  --dataset  bankmarketing --modelname tabtransformer --fold 5 >tabtransformer_bankmarketing_fold5.log&

nohup python train_classification.py  --dataset  bankmarketing --modelname BasicNet --fold 1 >mlp_bankmarketing_fold1.log&
nohup python train_classification.py  --dataset  bankmarketing --modelname BasicNet --fold 2 >mlp_bankmarketing_fold2.log&
nohup python train_classification.py  --dataset  bankmarketing --modelname BasicNet --fold 3 >mlp_bankmarketing_fold3.log&
nohup python train_classification.py  --dataset  bankmarketing --modelname BasicNet --fold 4 >mlp_bankmarketing_fold4.log&
nohup python train_classification.py  --dataset  bankmarketing --modelname BasicNet --fold 5 >mlp_bankmarketing_fold5.log&

nohup python train_classification.py  --dataset  bankmarketing --modelname kan --fold 1 >kan_bankmarketing_fold1.log&
nohup python train_classification.py  --dataset  bankmarketing --modelname kan --fold 2 >kan_bankmarketing_fold2.log&
nohup python train_classification.py  --dataset  bankmarketing --modelname kan --fold 3 >kan_bankmarketing_fold3.log&
nohup python train_classification.py  --dataset  bankmarketing --modelname kan --fold 4 >kan_bankmarketing_fold4.log&
nohup python train_classification.py  --dataset  bankmarketing --modelname kan --fold 5 >kan_bankmarketing_fold5.log&

nohup python tabnet_binary.py  --dataset  bankmarketing  --fold 1 >tabnet_bankmarketing_fold1.log&
nohup python tabnet_binary.py  --dataset  bankmarketing  --fold 2 >tabnet_bankmarketing_fold2.log&
nohup python tabnet_binary.py  --dataset  bankmarketing  --fold 3 >tabnet_bankmarketing_fold3.log&
nohup python tabnet_binary.py  --dataset  bankmarketing  --fold 4 >tabnet_bankmarketing_fold4.log&
nohup python tabnet_binary.py  --dataset  bankmarketing  --fold 5 >tabnet_bankmarketing_fold5.log&

nohup python xgboost_binary.py  --dataset  bankmarketing  --fold 1 >xgb_bankmarketing_fold1.log&
nohup python xgboost_binary.py  --dataset  bankmarketing  --fold 2 >xgb_bankmarketing_fold2.log&
nohup python xgboost_binary.py  --dataset  bankmarketing  --fold 3 >xgb_bankmarketing_fold3.log&
nohup python xgboost_binary.py  --dataset  bankmarketing  --fold 4 >xgb_bankmarketing_fold4.log&
nohup python xgboost_binary.py  --dataset  bankmarketing  --fold 5 >xgb_bankmarketing_fold5.log&

nohup python catboost_binary.py  --dataset  bankmarketing  --fold 1 >cat_bankmarketing_fold1.log&
nohup python catboost_binary.py  --dataset  bankmarketing  --fold 2 >cat_bankmarketing_fold2.log&
nohup python catboost_binary.py  --dataset  bankmarketing  --fold 3 >cat_bankmarketing_fold3.log&
nohup python catboost_binary.py  --dataset  bankmarketing  --fold 4 >cat_bankmarketing_fold4.log&
nohup python catboost_binary.py  --dataset  bankmarketing  --fold 5 >cat_bankmarketing_fold5.log&
