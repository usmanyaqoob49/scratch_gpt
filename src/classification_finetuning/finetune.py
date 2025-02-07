"""
This module has the function to finetune the gpt-2 model on the classificatio dataset.
"""
import torch
torch.manual_seed(123)
from .utils import batch_classification_loss

def finetune_model(model, train_loader, validation_loader, optimizer, device, num_epochs, eval_frequency, eval_iter):
    examples_seen= 0
    global_step= -1
    training_loss, validation_loss= [], []
    training_accuray, validation_accuracy= [], []

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            batch_loss= batch_classification_loss(gpt_model= model,
                                                  input_batch= input_batch,
                                                  target_batch= target_batch,
                                                  device= device)
            batch_loss.backward()
            optimizer.step()
            examples_seen+= input_batch.shape[0]


    

