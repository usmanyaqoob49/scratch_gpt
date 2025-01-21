"""
This module has implementation of transformer block.

We will use all the components like Multihead attention, LayerNorm, FeedForward to build this Transformer Block.
"""
import torch
import torch.nn as nn
from src.attention_mechanism.multihead_attention import MultiHeadAttention
from src.gpt.utils import FeedForward
from src.gpt.layer_norm import LayerNorm

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

        self.ff= FeedForward(cfg= cfg)

        self.LayerNorm1= 
    def forward(self, x):
        return x