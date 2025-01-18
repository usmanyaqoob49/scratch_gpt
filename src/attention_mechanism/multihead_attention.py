"""
Simply multihead attention is extension of single causal self-attention, 
In mulit-head, mulitple single causal attention instances are combined together each with its own weights.

Better to understand complex pattern, each head try to look at seprate angle of the input.

As we will have multiple context vecots (one from each head, we will concatenate them in the end).
"""
import torch
import torch.nn as nn

