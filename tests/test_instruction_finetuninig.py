import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.intruction_finetuning.utils import InstructionDataset, custom_collate
from src.data_preparation.utils import gpt_tokenizer
import json

#----------Testing Data Preparation Class
with open('./data/raw/instructions_data/instruction-data.json', 'r', encoding= 'utf-8') as file:
    data= json.load(file)
print('First data sample from the instructions dataset: ', data[0])
batch = [
    data[0],
    data[1],
]
gpt_2_tokenizer= gpt_tokenizer()
instruction_dataset= InstructionDataset(data= batch, tokenizer= gpt_2_tokenizer)
encoded_batch= instruction_dataset.encoded_text
print("Encoded: ", encoded_batch)
print('Number of encoded data samples: ', instruction_dataset.__len__())
print('-'*50)

#---------Testing Custom Collate functiont that pads the examples and make input and target samples from batch
input_tensors, target_tensors= custom_collate(batch= encoded_batch)
print('Input Tensors: ', input_tensors)
print('Output Tensors: ', target_tensors)
print('-'*50)
