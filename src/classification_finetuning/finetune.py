"""
This module has the function to finetune the gpt-2 model on the classificatio dataset.
"""
import torch
torch.manual_seed(123)

def classification_finetune(gpt_model, num_classes, got_configurations):
    for params in gpt_model.parameters():
        params.required_grad()= False
      