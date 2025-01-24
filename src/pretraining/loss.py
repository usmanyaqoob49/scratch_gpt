"""
Module that will compute loss so that we can do evaluation and pretrain according to reducing that loss.
"""
import torch
def compute_loss(model, inputs, targets):
    with torch.no_grad():
        logits= model(inputs)
    probabilities= torch.softmax(logits,
                                 dim= -1)
    