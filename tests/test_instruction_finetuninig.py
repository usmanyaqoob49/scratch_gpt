import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.intruction_finetuning.data_loader import create_istructions_data_loader
from src.data_preparation.utils import gpt_tokenizer
from src.
import json

#----------Testing Data Loader
with open('./data/raw/instructions_data/instruction-data.json', 'r', encoding= 'utf-8') as file:
    data= json.load(file)
print('First data sample from the instructions dataset: ', data[0])
batch = [
    data[0],
    data[1],
]
gpt_2_tokenizer= gpt_tokenizer()
training_loader, validation_loader, test_loader= create_istructions_data_loader(training_dataset=)