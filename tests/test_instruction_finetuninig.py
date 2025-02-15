import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.intruction_finetuning.data_loader import create_istructions_data_loader
from src.data_preparation.utils import gpt_tokenizer
from src.intruction_finetuning.utils import format_input
from src.intruction_finetuning.utils import train_val_test_split
from src.classification_finetuning.utils import get_gpt_2_openai
from src.pretraining.utils import text_to_tokens, tokens_to_text, generate_text, gpt_2_124m_configurations
import torch
from src.intruction_finetuning.train import train_model
import json

#----------Testing Data Loader
with open('./data/raw/instructions_data/instruction-data.json', 'r', encoding= 'utf-8') as file:
    data= json.load(file)
gpt_2_tokenizer= gpt_tokenizer()
train_set, validation_set, test_set= train_val_test_split(json_data= data)
training_loader, validation_loader, test_loader= create_istructions_data_loader(training_dataset=train_set,
                                                                                validation_dataset= validation_set,
                                                                                test_dataset= test_set,
                                                                                tokenizer= gpt_2_tokenizer,
                                                                                batch_size= 8,
                                                                                num_workers= 0)

#------Using the function that initialize the gpt-architecture and then loads openai weights in it
openai_gpt_2_model= get_gpt_2_openai()
openai_gpt_2_model.eval()

#------Testing not finetuned model on the instruction data
formatted_input_sample= format_input(input_entry=data[0])
print('First text data point foramatted in aplaca template: ', formatted_input_sample)
token_ids= generate_text(
    gpt_model= openai_gpt_2_model,
    idx= text_to_tokens(tokenizer= gpt_2_tokenizer,
                        text= formatted_input_sample),
    max_new_tokens= 50,
    context_size= gpt_2_124m_configurations['context_length']
)
generated_text= tokens_to_text(tokenizer= gpt_2_tokenizer, 
                                tokens_ids= token_ids)
print("Text produce by un-finetuned Gpt-2: ", generated_text)

#-----Testing training function for finetuning 
optim= torch.optim.AdamW(params= openai_gpt_2_model.parameters(),
                         lr= 0.00005,
                         weight_decay= 0.1)
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
instruction_finetuned_model, training_loss, validation_loss, track_tokens_seen= train_model(
    model= openai_gpt_2_model,
    train_loader= training_loader,
    validation_loader= validation_loader,
    optimizer=optim,
    device= device,
    eval_freq= 5,
    eval_iter= 5,
    start_context= formatted_input_sample,
    tokenizer= gpt_2_tokenizer
)
print('Training Loss for the Instruction finetuning: ', training_loss)
print('Validation Loss Loss for the Instruction finetuning: ', validation_loss)
