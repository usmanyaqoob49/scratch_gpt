import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.classification_finetuning.utils import balance_dataset

data= './data/raw/emotion_dataset/train.txt'
balance_data= balance_dataset(dataset_path= data, classes_column_name)