"""
This module has function to pretrain the LLM.
"""
import torch
from .utils import calculate_batch_loss
def train_model(model, train_loader, validation_loader,
                optimizer, device, num_epochs,
                eval_freq, eval_iter, start_context,
                tokenizer):
    training_loss, validattion_loss, track_tokens_seen= [], [], []
    tokens_seen, global_step= 0, -1
    
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            batch_loss= calculate_batch_loss(input_batch= input_batch,
                                             target_batch= target_batch,
                                             model= model,
                                             device= device)
            batch_loss.backward()
            optimizer.step()