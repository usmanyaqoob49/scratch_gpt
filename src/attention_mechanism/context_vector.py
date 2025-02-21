"""
This module finds the context vector.
"""
import torch

"""
This function will find the context vector of specific input from all inputs.
inputs----> All tokens 
query_index----> input of which we want to find context vector (that will show relation of that input will all others)
"""
def find_context_vector_query(inputs, query_index):
    query= inputs[query_index]
    
    attention_scores= torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        attention_scores[i]= torch.dot(query, x_i)

    attention_scores_normalized= torch.softmax(attention_scores, dim= 0)    #we have only one row so we can do row wise softmax

    context_vector_query= torch.zeros(size= query.shape)
    #context_vector= input*attension_score_normalized and summing
    for i, x_i in enumerate(inputs):
        context_vector_query+= attention_scores_normalized[i] * x_i
    return context_vector_query


"""
This function will find context vectors for all the input, that means relation of every token with every other token.
"""
def find_context_vector(inputs):
    attention_scores= torch.empty(inputs.shape[0], inputs.shape[0])
    for i, x_i in enumerate(inputs):    #take row
        for j, x_j in enumerate(inputs): #take row
            attention_scores[i][j]= torch.dot(x_i, x_j)

    normalized_attention_scores= torch.softmax(attention_scores, dim= -1)    #each tensor showld some to 1, so column wise softmax

    context_vector= normalized_attention_scores @ inputs  #matrix multiplication
    return context_vector