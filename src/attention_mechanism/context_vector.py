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
    