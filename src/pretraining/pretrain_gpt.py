"""
Module that has function to pretrain the GPT-2 model using the training function in train.py-
"""
import torch
from .train import train_model
from src.gpt.gpt_model import GPTModel
from src.gpt.utils import GPT_CONFIG_124M
from .train import train_model
from src.data_preparation.data_loader import create_data_loader_v1 
from src.data_preparation.utils import read_txt_file, gpt_tokenizer
from .utils import make_train_validation_loader

torch.manual_seed(123)

def pretrain_gpt(file_path, num_epochs):
    device= {
        'cuda' if torch.cuda.is_available() else 'cpu'
    }
    gpt_model= GPTModel(cfg= GPT_CONFIG_124M)
    gpt_model.to(device)
    optimizer= torch.optim.AdamW(
        gpt_model.parameters,
        lr= 0.0004,
        weight_decay= 0.1
    )

    text_data= read_txt_file(path= file_path)
    training_loader, validation_loader= make_train_validation_loader(
        text_data= text_data,
        train_data_ratio= 0.9
    )
    train_losses, validation_losses, tokens_seen= train_model(
        model= gpt_model,
        train_loader= training_loader,
        validation_loader= validation_loader,
        optimizer= optimizer,
        device= device,
        num_epochs= num_epochs,
        eval_freq= 5,
        eval_iter= 5,
        start_context= "Every effort moves you",
        tokenizer= gpt_tokenizer()
    )
    return train_losses, validation_losses, tokens_seen