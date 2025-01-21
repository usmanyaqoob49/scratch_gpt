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

        self.attention= MultiHeadAttention(
            d_in= cfg['emb_dim'],
            d_out= cfg['emb_dim'],
            context_length= cfg['context_length'],
            dropout= cfg['drop_out'],
            num_heads= cfg['n_heads'],
            qkv_bias= cfg['qkv_bias']
        )

        

    def forward(self, x):
        return x