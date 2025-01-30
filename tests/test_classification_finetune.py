import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.classification_finetuning.utils import balance_dataset, class_mapping, train_val_test_split, map_classes
import pandas as pd
from src.classification_finetuning.data_loader import create_data_loaders
from src.data_preparation.utils import gpt_tokenizer

data_path= './data/processed/emotion_dataset/combined_emotions_data.csv'
balance_data= balance_dataset(dataset_path= data_path, classes_column_name= 'emotion')
print(balance_data['emotion'].value_counts())
print('-'*50)

#------Testing random split function
data_df= pd.read_csv(data_path)
train_set, validation_set, test_set= train_val_test_split(data=data_df)
print('shape of train set: ', train_set.shape)
print('shape of validation set: ', validation_set.shape)
print('shape of test set: ', test_set.shape)
print('-'*50)

#-------Testing data loader class
train_dataset_loader= create_data_loaders(
    dataset= train_set,
    text_col_name= 'sentence',
    class_col_name= 'emotion',
    batch_size= 8,
    num_workers= 0,
    shuffle= True,
    drop_last= True
)
validation_dataset_loader= create_data_loaders(
    dataset=validation_set,
    text_col_name= 'sentence',
    class_col_name= 'emotion',
    batch_size= 8,
    num_workers= 0,
    shuffle= False,
    drop_last= False
    )
test_dataset_loader= create_data_loaders(
    data_df= test_set,
    text_col_name= 'sentence',
    labels_col_name= 'emotion',
    batch_size= 8,
    num_workers= 0,
    shuffle= False,
    drop_last= False)

print("Train loader:")
for input_batch, target_batch in train_dataset_loader:
    pass

print("Input batch dimensions:", input_batch.shape)
print("Label batch dimensions", target_batch.shape)