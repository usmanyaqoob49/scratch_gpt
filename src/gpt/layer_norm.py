"""
This module has class for implementation of layer normalization.
"""
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps= 1e-5):
        super().__init__()
    