"""
This module has implementation of transformer block.

We will use all the components like Multihead attention, LayerNorm, FeedForward to build this Transformer Block.

Here is what happens in transformer block:
#step 1---> Normalized
#step 2---> Multihead attention
#step 3---> Dropout
#step 4---> Add shortcut connection (add input to output)
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

        self.norm1= LayerNorm(emb_dims= cfg['emb_dim'])
        self.norm2= LayerNorm(emb_dims= cfg['emb_dim'])

        self.drop_shortcut= nn.Dropout(p= cfg['drop_out'])
    def forward(self, x):
        shortcut_connection= x

        normalized1= self.norm1(x)
        context_vector= self.attention(normalized1)
        dropped_out= self.drop_shortcut(context_vector)
        out= dropped_out + shortcut_connection

        shortcut_connection= out        #update the shortcut connection with output of block1
        normalized2= self.norm2(out)
        ff_out= self.ff(normalized2)
        

        return x