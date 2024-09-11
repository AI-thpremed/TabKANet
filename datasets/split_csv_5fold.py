import pandas as pd
import os


# 6:2:2
data = pd.read_csv("/data/gaowh/work/24process/TabKANet/datasets/onlineshoper/online_shoppers_intention.csv")
data = data.sample(frac=1).reset_index(drop=True)

key="onlineshoper"

for i in range(1, 6):
    os.makedirs(f"/data/gaowh/work/24process/TabKANet/datasets/"+key+"/Fold{i}", exist_ok=True)

test_size = len(data) // 5
for i in range(5):
    test_data = data.iloc[i*test_size:(i+1)*test_size]
    test_data.to_csv(f"/data/gaowh/work/24process/TabKANet/datasets/{key}/Fold{i+1}/test.csv", index=False)

    # Calculate remaining data excluding the test data
    remaining_data = pd.concat([data.iloc[:i*test_size], data.iloc[(i+1)*test_size:]])

    train_size = int(6 * len(remaining_data)/8)
    val_size = len(remaining_data) - train_size
    train_data = remaining_data.iloc[:train_size]
    val_data = remaining_data.iloc[train_size:]

    # Save train and val data for each fold
    train_data.to_csv(f"/data/gaowh/work/24process/TabKANet/datasets/{key}/Fold{i+1}/train.csv", index=False)
    val_data.to_csv(f"/data/gaowh/work/24process/TabKANet/datasets/{key}/Fold{i+1}/val.csv", index=False)
