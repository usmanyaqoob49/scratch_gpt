import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.classification_finetuning.utils import balance_dataset

data= './data/processed/emotion_dataset/train.csv'
balance_data= balance_dataset(dataset_path= data, c)