"""
This module has class for implementation of layer normalization.

Purpose of Layer Normalization is to adjust output of previous layer in the way that they should have
0 mean and 1 variance. Helps in convergence of to effective weights and effiecient training.
"""
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps= 1e-5):
        super().__init__()
    
    def forward(self, x):
        return x