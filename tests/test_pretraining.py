"""
Test cases for different modules of pretraining.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pretraining.utils import text_to_tokens, tokens_to_text
from src.gpt.gpt_model import GPTModel
from src.gpt.utils import GPT_CONFIG_124M
from src.data_preparation.utils import gpt_tokeinzer
from src.gpt.utils import generate_text
from src.data_preparation.utils import read_verdict
from src.data_preparation.data_loader import create_data_loader_v1 
import torch
torch.manual_seed(123)

#-----------Testing text to token and token to text functions
tokenizer= gpt_tokeinzer()
gpt_model= GPTModel(cfg= GPT_CONFIG_124M)
start_context = "Every effort moves you"
token_ids_result= generate_text(
    gpt_model= gpt_model,
    idx= text_to_tokens(tokenizer= tokenizer,
                        text= start_context),
    max_new_tokens= 10,
    context_size= GPT_CONFIG_124M['context_length'],
)
print('Output of GPT Model: ', tokens_to_text(tokenizer= tokenizer,
                                              tokens_ids= token_ids_result))
print("---------------------------------------------------")

#-----------Testing loss functions
text_data= read_verdict(path= "../data/raw/the-verdict.txt")
train_data_ratio= 0.90
train_data_text= text_data[:len(text_data) * train_data_ratio]
validation_data_text= text_data[len(text_data) * train_data_ratio:]

train_loader= create_data_loader_v1(
    txt= train_data_text,
    batch_size= 2,
    max_length= GPT_CONFIG_124M['context_length'],
    stride= GPT_CONFIG_124M['context_length'],
    drop_last= True,
    shuffle= True,
    num_workers= 0
)

validation_loader= create_data_loader_v1(
    txt= validation_data_text,
    batch_size= 2,
    max_length= GPT_CONFIG_124M['context_length'],
    stride= GPT_CONFIG_124M['context_length'],
    drop_last= False,
    shuffle= False,
    num_workers= 0
)
print('Train Loader: ')
for input, target in train_loader:
    print(input.shape, target.shape)
print('Validataion Loader: ')
for input, target in validation_loader:
    print(input.shape, target.shape)
    