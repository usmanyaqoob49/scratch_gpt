"""
Simply multihead attention is extension of single causal self-attention, 
In mulit-head, mulitple single causal attention instances are combined together each with its own weights.

Better to understand complex pattern, each head try to look at seprate angle of the input.

As we will have multiple context vecots (one from each head, we will concatenate them in the end).

Remember d_out which is context vector (col) number, must be divisible by num_heads so that each head
can have same size context vector.
"""
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias= False):
        super().__init__()

        assert (d_out % num_heads == 0),    \
            "d_out must be divisible by num_heads so that each head can have same sized context vector."