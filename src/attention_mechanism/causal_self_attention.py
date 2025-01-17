"""
In standard self attention that is implemented in self_attention.py, while finding the context vector,
it has access of all the attention weights current, previous and future.

But in causal self attention future tokens are masked out and not shown,
only current and previous tokens are processed together, this is done by masking out the attention
weights of the future tokens.

We will also have drop out layer implemented in this to avoid overfitting.
"""
import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias= False):
        super().__init__()

        self.W_query= nn.Linear(d_in, d_out, bias= qkv_bias)
        self.W_key= nn.Linear(d_in, d_out, bias= qkv_bias)
        self.W_value= nn.Linear(d_in, d_out, bias= qkv_bias)

        self.drop_out= nn.Dropout(p= dropout)
        self.d_out= d_out

        #for masking upper diagonal (future values)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        num_example, num_tokens, d_in= x.shape
        #step 1-->Find the key, query, value
        keys= self.W_key(x)
        queries= self.W_query(x)
        value= self.W_value(x)

        #step 2-->Find the attention weights
        attention_weights= queries @ keys.Transpose(1,2)    #as we pass multiple examples (batch) so we will take transpose w.r.t row and col
        
        #step 3-->Mask the future values with -inf (so that when softmax applied they become zero)
        masked_attention_weights= attention_weights.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )

        #step 4-->Normalize the masked attention weights
        masked_attention_scores= torch.softmax(
            masked_attention_weights / keys.shape[-1]**0.5, dim=-1
        )
