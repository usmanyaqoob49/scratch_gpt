"""
Simply multihead attention is extension of single causal self-attention, 
In mulit-head, mulitple single causal attention instances are combined together each with its own weights.

Better to understand complex pattern, each head try to look at seprate angle of the input.

As we will have multiple context vecots (one from each head, we will concatenate them in the end).
"""
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias= Flase):
        super().__init__()
        