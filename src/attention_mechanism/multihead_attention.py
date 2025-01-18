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
        self.d_out= d_out
        self.num_head= num_heads
        self.head_dim= d_out // num_heads              #dimentions of each heads context vector d_out/total heads
        
        self.W_query= nn.Linear(d_in, d_out, bias= qkv_bias)
        self.W_key= nn.Linear(d_in, d_out, bias= qkv_bias)
        self.W_value= nn.Linear(d_in, d_out, bias= qkv_bias)

        self.out_proj= nn.Lienar(d_out, d_out)  #to combine heads outputs
        self.drop_out= nn.Dropout(p= dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),  diagonal= 1)
        )
        