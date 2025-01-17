"""
In standard self attention that is implemented in self_attention.py, while finding the context vector,
it has access of all the attention weights current, previous and future.

But in causal self attention future tokens are masked out and not shown,
only current and previous tokens are processed together, this is done by masking out the attention
weights of the future tokens.

We will also have drop out layer implemented in this to avoid overfitting.
"""
import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias= False):
        super().__init__()

        self.W_query= nn.Linear(d_in, d_out, bias= qkv_bias)
        self.W_key= nn.Linear(d_in, d_out, bias= qkv_bias)
        self.W_value= nn.Linear(d_in, d_out, bias= qkv_bias)

    def forward(self, x):
        #step 1-->Find the key, query, value
        keys= self.W_key(x)
        query= self.W_query(x)
        value= self.W_value(x)

        