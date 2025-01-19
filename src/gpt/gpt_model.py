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

    