"""
This module has the implementation of the self attention class.---> Self attention is finding relation with in the input (self)

Main purpose of self attention is to find the context vector for each input.
Each input will have its on context vector that will have info of that input and all other inputs.

We find it using attention weights and inputs. (Matrix Multiplication)
"""
import torch

class selfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias= False):
        
        