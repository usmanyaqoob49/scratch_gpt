import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.classification_finetuning.utils import balance_dataset, class_mapping, train_val_test_split
import pandas as pd

data= './data/processed/emotion_dataset/combined_emotions_data.csv'
balance_data= balance_dataset(dataset_path= data, classes_column_name= 'emotion')
print(balance_data['emotion'].value_counts())

#------Testing class lable mapping function
labels_mapping= class_mapping(classes_list= balance_data['emotion'].unique().tolist())
print('Lable mapping for the dataset: ', labels_mapping)

#------Testing random split function
data_df= pd.read_csv(data_path)
train_set, validation_set, test_set= train_val_test_split(data=data)
print('shape of train set: ', train_set.shape)
print('shape of validation set: ', validation_set.shape)
print('shape of test set: ', test_set.shape)