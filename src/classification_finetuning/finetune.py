"""
This module has the function to finetune the gpt-2 model on the classificatio dataset.
"""
import torch

def classification_finetune(gpt_model, num_classes):
    for params in gpt_model.parameters():
        params.required_grad()= False