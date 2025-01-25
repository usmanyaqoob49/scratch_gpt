"""
Module to pretrain the GPT-2 model using the training function in train.py-
"""
import torch
from .train import train_model
from src.gpt.gpt_model import GPTModel
from src.gpt.utils import GPT_CONFIG_124M
from .train import train_model
from src.data_preparation.data_loader import create_data_loader_v1 

device= {
    'cuda' if torch.cuda.is_available() else 'cpu'
}
torch.manual_seed(123)
gpt_model= GPTModel(cfg= GPT_CONFIG_124M)
gpt_model.to(device)
optimizer= torch.optim.AdamW(
    gpt_model.parameters,
    lr= 0.0004,
    weight_decay= 0.1
)
num_epochs= 10

train_losses, validation_losses, tokens_seen= train_model(
    model= gpt_model,
    train_loader= 
)