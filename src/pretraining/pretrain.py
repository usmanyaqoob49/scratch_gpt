"""
Module to pretrain the GPT-2 model using the training function in train.py-
"""
import torch
from .train import train_model
from src.gpt.gpt_model import GPTModel
from src.gpt.utils import GPT_CONFIG_124M

device= {
    'cuda' if torch.cuda.is_available() else 'cpu'
}
torch.manual_seed(123)
gpt_model= GPTModel(cfg= GPT_CONFIG_124M)
gpt_model.to(device)
