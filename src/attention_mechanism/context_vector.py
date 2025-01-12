"""
This module finds the context vector for each input.
"""

"""
inputs----> All tokens 
query_index----> input of which we want to find context vector (that will show relation of that input will all others)
"""
import torch

def find_context_vector(inputs, query_index):
    query= inputs[query_index]
    
    attention_scores= torch.empty(size= inputs.shape[0])
    for i, x_i in enumerate(inputs):
        attention_scores[i]= torch.dot(query, x_i)

    attention_scores_normalized= attention_scores/attention_scores.sum()