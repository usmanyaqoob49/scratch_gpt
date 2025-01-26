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
from src.pretraining.utils import calculate_loader_loss, make_train_validation_loader, load_gpt2_params_from_tf_ckpt, gpt_2_124m_configurations
from src.pretraining.pretrain_gpt import pretrain_gpt
from src.pretraining.generate_text import generate_diverse
from src.pretraining.loading_weight_into_gpt import loads_weight_into_gpt
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
print('Pretraining GPT-2')
# training_loss, validation_loss, tokens_seen= pretrain_gpt(
#     file_path= './data/raw/the-verdict.txt',
#     num_epochs= 10
# )
# print('***********After training**************')
# print('Training Loss: ', training_loss)
# print('Validation Loss: ', validation_loss)
# print('Num of Token seen in training: ', tokens_seen)
print("---------------------------------------------------")


#-----------Testing the creative text generation function
token_ids= generate_diverse(
    model= gpt_model,
    idx= text_to_tokens(tokenizer= gpt_tokenizer(), text= "Every effort moves you"),
    max_new_tokens= 15,
    context_size= GPT_CONFIG_124M['context_length'],
    top_k= 25,
    temperature= 1.4
)
print("Diverse output: ", tokens_to_text(tokenizer= gpt_tokenizer(), tokens_ids= token_ids))
print("---------------------------------------------------")


#-----------Testing the creative text generation function
print('Loading weights of openAI in gpt-2 architecture that we have created and testing its text generation: ')
openai_parameters= load_gpt2_params_from_tf_ckpt(ckpt_path= './Model/gpt-2/124M',
                                                 settings= gpt_2_124m_configurations)
try:
    openai_gpt_model= loads_weight_into_gpt(gpt_model= gpt_model,
                                        params= openai_parameters)
    print("Weights loaded successfully in GPT-2 architecture!")
except:
    raise ValueError("Failed Loading of Weights!")
