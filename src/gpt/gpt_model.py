"""
It has dummy gpt model class.
"""
import torch
import torch.nn as nn
from src.data_preparation.utils import gpt_tokeinzer
from layer_norm import LayerNorm
from src.transformer.transformer import Transformer

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        #token embeddings, position embeddings, and dropout
        self.token_emb= nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.positional_emb= nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_out= nn.Dropout(cfg['drop_out'])

        #multiple transforners blocks
        self.transformer_blocks= [
            *[Transformer(cfg=cfg) for _ in range(cfg['n_layers'])]
        ]

        #layer normalization
        self.final_norm= LayerNorm(cfg['emb_dim'])

        self.out_head= nn.Linear(
            cfg['emb_dim'], cfg['vocab_size'], bias= False
        )
    
    def forward(self, in_idx):
        batch_size, seq_len= in_idx.shape
        token_emb= self.token_emb(in_idx)
        position_emb= self.positional_emb(torch.arange(seq_len, device= in_idx.device()))
                                          