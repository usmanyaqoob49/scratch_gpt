"""
This module has class for implementation of layer normalization.

Purpose of Layer Normalization is to adjust output of previous layer in the way that they should have
0 mean and 1 variance. Helps in convergence of to effective weights and effiecient training.

In the end we will scale and shift the normalized output by values obtained from trainable parameters 
to achieve best results. (These values are found during trianing process.)
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
        mean= x.mean(dim= -1, keepdim= True)
        variance= x.var(dim= -1, keepdim= True, unbiased= False) 
        normalized_output= (x-mean) / torch.sqrt(variance + self.eps) 
        return self.scale * normalized_output + self.shift