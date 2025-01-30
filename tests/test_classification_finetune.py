import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.classification_finetuning.utils import balance_dataset, class_mapping, train_val_test_split, map_classes
import pandas as pd
from src.classification_finetuning.data_loaders import ClassDataset
from src.data_preparation.utils import gpt_tokenizer

data_path= './data/processed/emotion_dataset/combined_emotions_data.csv'
balance_data= balance_dataset(dataset_path= data_path, classes_column_name= 'emotion')
print(balance_data['emotion'].value_counts())
print('-'*50)

#------Testing class lable mapping function
labels_mapping= class_mapping(classes_list= balance_data['emotion'].unique().tolist())
print('Lable mapping for the dataset: ', labels_mapping)
print('-'*50)

#------Testing random split function
data_df= pd.read_csv(data_path)
train_set, validation_set, test_set= train_val_test_split(data=data_df)
print('shape of train set: ', train_set.shape)
print('shape of validation set: ', validation_set.shape)
print('shape of test set: ', test_set.shape)
print('-'*50)

#Testing class mapping function
mapped_train_set= map_classes(dataset= train_set, 
                             class_col_name= 'emotion',
                             class_mapping_dict= labels_mapping)
mapped_validation_set= map_classes(dataset= validation_set, 
                             class_col_name= 'emotion',
                             class_mapping_dict= labels_mapping)
mapped_test_set= map_classes(dataset= test_set, 
                             class_col_name= 'emotion',
                             class_mapping_dict= labels_mapping)

#-------Testing data loader class
train_dataset_loader= ClassDataset(data_df= train_set,
                                   text_col_name= 'sentence',
                                   labels_col_name= 'emotion',
                                   tokenizer= gpt_tokenizer())
print('Train dataset sample: ', train_dataset_loader.__getitem__(2))