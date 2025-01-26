"""
This module has the function that loads the openAI's weights in gpt arhcitecure that we scratch coded.
"""
import torch
import numpy as np
from .utils import assign

def loads_weight_into_gpt(gpt_model, params):
    #positional and token embeddings weights
    gpt_model.pos_emb.weights= assign(gpt_model.pos_emb.weights, params['wpe'])
    gpt_model.tok_emb.weights= assign(gpt_model.tok_emb.weights, params['wte'])

    for b in range(len(params['blocks'])):
        #q,k,v weights
        q_w, k_w, v_w= np.split((params['blocks'][b]['attn']['c_attn'])['w'], 3, axis= -1)
        gpt_model.trf_blocks[b].att.W_query.weight= assign(
            gpt_model.trf_block[b].att.W_query.weight, q_w.T
        )
        gpt_model.trf_blocks[b].att.W_key.weight= assign(
            gpt_model.trf_block[b].att.W_key.weight, k_w.T
        )
        gpt_model.trf_blocks[b].att.W_value.weight= assign(
            gpt_model.trf_block[b].att.W_value.weight, v_w.T
        )
        #q,k,v biases
        q_b, k_b, v_b= np.split((params['block'][b]['attn']['c_attn'])['b'], 3, axis=-1)
        

