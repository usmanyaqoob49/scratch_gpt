"""
This module has the function to finetune the gpt-2 model on the classificatio dataset.
"""
import torch
torch.manual_seed(123)

def classification_finetune(gpt_model, num_classes, got_configurations):
    for params in gpt_model.parameters():
        params.required_grad()= False
    gpt_model.out_head= torch.nn.Linear(in_features= got_configurations['emb_dim'],
                                         out_features= num_classes)
    for params in gpt_model.out_head.parameters():
        params.requires_grad()= True
    for params in gpt_model.final_norm.parameters():
        params.requires_grad()= True