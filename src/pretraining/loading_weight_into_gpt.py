"""
This module has the function that loads the openAI's weights in gpt arhcitecure that we scratch coded.
"""
import torch
import numpy as np
from .utils import assign

def loads_weight_into_gpt(gpt_model, params):
    gpt_model.pos_emb.weights= assign(gpt_model.pos_emb.weights, params['wpe'])

