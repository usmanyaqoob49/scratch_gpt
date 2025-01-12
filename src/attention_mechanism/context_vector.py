"""
This module finds the context vector.
"""
import torch

"""
This function will find the context vector of specific input from all inputs.
inputs----> All tokens 
query_index----> input of which we want to find context vector (that will show relation of that input will all others)
"""
def find_context_vector(inputs, query_index):
    query= inputs[query_index]
    
    attention_scores= torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        attention_scores[i]= torch.dot(query, x_i)

    attention_scores_normalized= torch.softmax(attention_scores, dim= 0)

    context_vector_query= torch.zeros(size= query.shape)
    #context_vector= input*attension_score_normalized and summing
    for i, x_i in enumerate(inputs):
        context_vector_query+= attention_scores_normalized[i] * x_i
    return context_vector_query

