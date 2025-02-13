import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.intruction_finetuning.utils import InstructionDataset, custom_collate
from src.data_preparation.utils import gpt_tokenizer
import json

with open('./data/raw/instructions_data/instruction-data.json', 'r', encoding= 'utf-8') as file:
    data= json.load(file)

print('First data sample from the instructions dataset: ', data[0])
batch = [
    data[0],
    data[1],
]
gpt_2_tokenizer= gpt_tokenizer()
instruction_dataset= InstructionDataset(data= batch, tokenizer= gpt_2_tokenizer)
print("Encoded: ", instruction_dataset.encoded_text)


