"""
This module has the function that loads the openAI's weights in gpt arhcitecure that we scratch coded.
"""
import torch
import numpy as np
from .utils import assign

def loads_weight_into_gpt(gpt_model, params):
    #positional and token embeddings weights
    gpt_model.positional_emb.weight= assign(gpt_model.positional_emb.weight, params['wpe'])
    gpt_model.token_emb.weight= assign(gpt_model.token_emb.weight, params['wte'])

    for b in range(len(params['blocks'])):
        #q,k,v weights
        q_w, k_w, v_w= np.split((params['blocks'][b]['attn']['c_attn'])['w'], 3, axis= -1)
        gpt_model.transformer_blocks[b].attention_scores.W_query.weight= assign(
            gpt_model.transformer_blocks[b].attention_scores.W_query.weight, q_w.T
        )
        gpt_model.transformer_blocks[b].attention_scores.W_key.weight= assign(
            gpt_model.transformer_blocks[b].attention_scores.W_key.weight, k_w.T
        )
        gpt_model.transformer_blocks[b].attention_scores.W_value.weight= assign(
            gpt_model.transformer_blocks[b].attention_scores.W_value.weight, v_w.T
        )
        #q,k,v biases
        q_b, k_b, v_b= np.split((params['block'][b]['attn']['c_attn'])['b'], 3, axis=-1)
        gpt_model.transformer_blocks[b].attention_scores.W_query.bias= assign(
            gpt_model.transformer_blocks[b].attention_scores.W_query.bias, q_b
        )
        gpt_model.transformer_blocks[b].attention_scores.W_key.bias= assign(
            gpt_model.transformer_blocks[b].attention_scores.W_key.bias, k_b
        )
        gpt_model.transformer_blocks[b].attention_scores.W_value.bias= assign(
            gpt_model.transformer_blocks[b].attention_scores.W_value.bias, v_b
        )
         #getting attention output projection weights and biases
        gpt_model.transformer_blocks[b].attention_scores.out_proj.weight = assign(
            gpt_model.transformer_blocks[b].attention_scores.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt_model.transformer_blocks[b].attention_scores.out_proj.bias = assign(
            gpt_model.transformer_blocks[b].att.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt_model.transformer_blocks[b].ff.layers[0].weight = assign(
            gpt_model.transformer_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt_model.transformer_blocks[b].ff.layers[0].bias = assign(
            gpt_model.transformer_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt_model.transformer_blocks[b].ff.layers[2].weight = assign(
            gpt_model.transformer_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt_model.transformer_blocks[b].ff.layers[2].bias = assign(
            gpt_model.transformer_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt_model.transformer_blocks[b].norm1.scale = assign(
            gpt_model.transformer_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt_model.transformer_blocks[b].norm1.shift = assign(
            gpt_model.transformer_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt_model.transformer_blocks[b].norm2.scale = assign(
            gpt_model.transformer_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt_model.transformer_blocks[b].norm2.shift = assign(
            gpt_model.transformer_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt_model.final_norm.scale = assign(gpt_model.final_norm.scale, params["g"])
    gpt_model.final_norm.shift = assign(gpt_model.final_norm.shift, params["b"])
    gpt_model.out_head.weight = assign(gpt_model.out_head.weight, params["wte"])

    return gpt_model