"""
This module has all utils like configuration of the gpt model, implementation of the GELU, implmentation of 
feed forward network module that all will be used in gpt module.
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

#------GeLu Implementation--->This is specific implementation use in GPT-2
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))