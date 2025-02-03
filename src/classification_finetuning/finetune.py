"""
This module has the function to finetune the gpt-2 model on the classificatio dataset.
"""
import torch
torch.manual_seed(123)

def finetune_model(model, train_loader, validation_loader, optimizer, device, num_epochs, eval_frequency, eval_iter):
    for epoch in num_epochs:
        for i, (input_batch, target_batch) in enumerate(train_loader):
            input_batch, target_batch= input_batch.to(device), target_batch.to(device)
            