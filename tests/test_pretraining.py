"""
Test cases for different modules of pretraining.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pretraining.utils import text_to_tokens, tokens_to_text
from src.gpt.gpt_model import GPTModel
from src.gpt.utils import GPT_CONFIG_124M
from src.data_preparation.utils import gpt_tokenizer
from src.gpt.utils import generate_text
from src.data_preparation.utils import read_txt_file
from src.data_preparation.data_loader import create_data_loader_v1 
from src.pretraining.utils import calculate_loader_loss, make_train_validation_loader
from src.pretraining.pretrain_gpt import pretrain_gpt
import torch
torch.manual_seed(123)

#-----------Testing text to token and token to text functions
tokenizer= gpt_tokenizer()
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
text_data= read_txt_file(path= "./data/raw/the-verdict.txt")
train_loader, validation_loader= make_train_validation_loader(text_data= text_data,
                                                              train_data_ratio= 0.9)
print("Train loader:")
print("Train loader length: ", len(train_loader))
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader:")
for x, y in validation_loader:
    print(x.shape, y.shape)
    
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpt_model.to(device)

with torch.no_grad():
    train_loss= calculate_loader_loss(data_loader= train_loader,
                                      model= gpt_model,
                                      device= device)
    validation_loss= calculate_loader_loss(data_loader= validation_loader,
                                           model= gpt_model,
                                           device= device)
print("Training Loss: ", train_loss)
print("Validation Loss: ", validation_loss)
print("---------------------------------------------------")


#-----------Pretraining GPT-2 on verdict dataset
training_loss, validation_loss, tokens_seen= pretrain_gpt(
    file_path= './data/raw/the-verdict.txt',
    num_epochs= 10
)
print('***********After training**************')
print('Training Loss: ', training_loss)
print('Validation Loss: ', validation_loss)
print('Num of Token seen in training: ', tokens_seen)