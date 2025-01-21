"""
This module has implementation of transformer block.

We will use all the components like Multihead attention, LayerNorm, FeedForward to build this Transformer Block.
"""
import torch
import torch.nn as nn
from src.attention_mechanism.multihead_attention import MultiHeadAttention

class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()


    def forward(self, x):
        return x