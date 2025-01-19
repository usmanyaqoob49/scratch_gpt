"""
This module has class for implementation of layer normalization.

Purpose of Layer Normalization is to adjust output of previous layer in the way that they should have
0 mean and 1 variance. Helps in convergence of to effective weights and effiecient training.
"""
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dims):
        super().__init__()
        self.eps= 1e-5  #very small number that we will add in denuminator to prevent division by 0 problem
        self.scale= nn.Parameter(torch.ones(emb_dims))
        self.shift= nn.Parameter(torch.zeros(emb_dims))
    
    def forward(self, x):
        means= torch.mean(x, dim= -1, keepdim= True)
        variance= torch.var(x, dim= -1, keepdim= True)
        
        return x