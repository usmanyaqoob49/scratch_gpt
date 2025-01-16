import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.attention_mechanism.context_vector import find_context_vector_query, find_context_vector
from src.attention_mechanism.self_attention import SelfAttention
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

self_attention_object= SelfAttention

