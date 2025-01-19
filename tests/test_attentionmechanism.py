import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.attention_mechanism.context_vector import find_context_vector_query, find_context_vector
from src.attention_mechanism.self_attention import SelfAttention
from src.attention_mechanism.causal_self_attention import CausalAttention
from src.attention_mechanism.multihead_attention import MultiHeadAttention
import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

context_vector_1= find_context_vector_query(inputs= inputs, query_index= 1)
print('Context vector for 2nd input (journey): ', context_vector_1)

#testing function that returns complete context vector of all the inputs
complete_context_vector= find_context_vector(inputs= inputs)
print(complete_context_vector)

#----Testing self attention
self_attention_object= SelfAttention(d_in= inputs.shape[1], d_out= 2)
context_vector= self_attention_object.forward(x= inputs)
print("Complete Scaled Self attention-Context Vector: ", context_vector)


#------Testing causal attention
batch= torch.stack((inputs, inputs), dim= 0)  #combining inputs to make batch
print('Batch for causal attention: ', batch)
causal_attention= CausalAttention(d_in= batch.shape[2], d_out= 2, context_length= batch.shape[1], dropout= 0.0)
context_vector_causal= causal_attention.forward(batch)
print('Context Vector of Causal Attention: ', context_vector_causal)


#------Testing multihead attention
num_example, num_tokens, d_in= batch.shape
d_out= 2
num_heads= 2
mulihead_attention= MultiHeadAttention(d_in= d_in, d_out= d_out, context_length= num_tokens, dropout= 0.0, num_heads= num_heads)
context_vector_mulithead= mulihead_attention.forward(x= batch)
print(context_vector_mulithead)