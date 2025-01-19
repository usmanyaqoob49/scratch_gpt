"""
This module has all utils like configuration of the gpt model, implementation of the GELU.
"""
import torch
import torch.nn as nn

#Settings for GPT-2 (124M) 
GPT_CONFIG_124M= {
    'vocab_size': 50257, 
    'context_length': 1024,
    'emb_dim': 768,
    'n_heads': 12,
    'n_layers': 12,
    'drop_out': 0.1, 
    'qkv_bias': False
}

#------GeLu Implementation
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    