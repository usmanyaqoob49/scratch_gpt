import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.classification_finetuning.utils import balance_dataset, class_mapping

data= './data/processed/emotion_dataset/train_set.csv'
balance_data= balance_dataset(dataset_path= data, classes_column_name= 'emotion')
print(balance_data['emotion'].value_counts())

#------Testing class lable mapping function
