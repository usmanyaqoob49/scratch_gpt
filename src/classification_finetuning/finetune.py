"""
This module has the function to finetune the gpt-2 model on the classificatio dataset.
"""
import torch
torch.manual_seed(123)

def finetune_model(model, train_loader, validation_loader, optimizer, device, num_epochs, eval_frequency, eval_iter):
    for epoch in num_epochs:
        model.train()
        for input_batch, target_batch in train_loader:
            input_batch, target_batch= input_batch.to(device), target_batch.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                
