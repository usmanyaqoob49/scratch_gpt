"""
It has implementation gpt model class.

This is what happens in gpt:
        input --->token embeddings --->Position embeddings ---> token emb + positional emb --->drop out --->transformer blocks --->layer norm --->logits

It will give us logits that will be the probability of all the tokens in the vocabulay to be the next token.
"""
import torch
import torch.nn as nn
from src.data_preparation.utils import gpt_tokenizer
from .layer_norm import LayerNorm
from src.transformer.transformer_block import Transformer

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        #token embeddings, position embeddings, and dropout
        self.token_emb= nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.positional_emb= nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_out= nn.Dropout(cfg['drop_out'])

        #multiple transforners blocks connexted sequentially
        self.transformer_blocks= nn.Sequential(
            *[Transformer(cfg=cfg) for _ in range(cfg['n_layers'])]
        )

        #layer normalization
        self.final_norm= LayerNorm(cfg['emb_dim'])

        #final linear layer that will give us logits for all tokens in vocabulary
        self.out_head= nn.Linear(
            cfg['emb_dim'], cfg['vocab_size'], bias= False
        )
    
    def forward(self, in_idx):
        batch_size, seq_len= in_idx.shape
        token_emb= self.token_emb(in_idx)
        position_emb= self.positional_emb(torch.arange(seq_len, device= in_idx.device))

        x= token_emb + position_emb
        x_drop_out= self.drop_out(x)
        x_trf_block= self.transformer_blocks(x_drop_out)
        x_norm= self.final_norm(x_trf_block)
        logits= self.out_head(x_norm)

        return logits

                                          