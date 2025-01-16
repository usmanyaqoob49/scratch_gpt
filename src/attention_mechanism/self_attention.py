"""
This module has the implementation of the self attention class.---> Self attention is finding relation with in the input (self)

Main purpose of self attention is to find the context vector for each input.
Each input will have its on context vector that will have info of that input and all other inputs.

We find it using attention weights and inputs. (Matrix Multiplication)
"""
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query= nn.Linear(
            d_in, d_out, bias= qkv_bias
        )
        self.W_key= nn.Linear(
            d_in, d_out, bias= qkv_bias
        )
        self.W_value= nn.Linear(
            d_in, d_out, bias= qkv_bias
        )

    def forward(self, x):
        #Find query, key, value
        query= x @ self.W_query
        key= x @ self.W_key
        value= x @ self.W_value

        #Find attention weights (unnormalized)
        attention_weights= query @ key.T

        #Unnormalizing the attention wrights--->scaled self attenion
        attention_scores= torch.softmax(attention_weights / key.shape[1] ** 5, dim= -1)

        #Finding context vector
        context_vector= attention_scores @ value
        return context_vector

        