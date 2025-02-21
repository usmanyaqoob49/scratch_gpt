"""
Simply multihead attention is extension of single causal self-attention, 
In mulit-head, mulitple single causal attention instances are combined together each with its own weights.

Better to understand complex pattern, each head try to look at seprate angle of the input.

As we will have multiple context vecots (one from each head, we will concatenate them in the end).

Remember d_out which is context vector (col) number, must be divisible by num_heads so that each head
can have same size context vector.

We will split input into multiple heads by reshaping our projected query, key, value and then combining
the results.
"""
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias= False):
        super().__init__()

        assert (d_out % num_heads == 0),    \
            "d_out must be divisible by num_heads so that each head can have same sized context vector."
        self.d_out= d_out
        self.num_heads= num_heads
        self.head_dim= d_out // num_heads              #dimentions of each heads context vector d_out/total heads
        
        self.W_query= nn.Linear(d_in, d_out, bias= qkv_bias)
        self.W_key= nn.Linear(d_in, d_out, bias= qkv_bias)
        self.W_value= nn.Linear(d_in, d_out, bias= qkv_bias)

        self.out_proj= nn.Linear(d_out, d_out)  #to combine heads outputs
        self.drop_out= nn.Dropout(p= dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),  diagonal= 1)
        )
        
    def forward(self, x):
        num_examples, num_tokens, d_in= x.shape

        keys= self.W_key(x)
        queries= self.W_query(x)
        values= self.W_value(x)

        #---now we have to reshape our keys, query, value to form mulitple heads
        """
        Before splitting if each keys, values, queries was like for num_example=1, num_tokens=3, d_out=2              
                    [1.1,2.1]
                    [1.2,2.2]
                    [1.3,2.3]   

        After applying reshaping it will become like:
                    [
                    [ [1.1],[2.1]]
                    [ [1.2],[2.2]]
                    [ [1.3],[2.3]]
                    ]
        Where now each token 1 d_out represent response of one head seprtely. I hope it makes sense:)
        """
        keys= keys.view(num_examples, num_tokens, self.num_heads, self.head_dim)
        values= values.view(num_examples, num_tokens, self.num_heads, self.head_dim)
        queries= queries.view(num_examples, num_tokens, self.num_heads, self.head_dim)

        #---Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim) to match the shapes for matrix mulitplication
        keys= keys.transpose(1,2)
        values= values.transpose(1,2)
        queries= queries.transpose(1,2)

        #---Find attention weights
        """
        query shape= (b, num_heads, num_tokens, head_dim), 
        after transpose key will be= (b, num_heads, head_dim,  num_tokens)
        so number of col of query (head_dim)= number of rows of keys (head_dim)
        
        shape of attention scores will be: (num_examples,num_heads,num_tokens,num_tokens)
        """
        attention_weights= queries @ keys.transpose(2,3)

        #mask the attention weights of future tokens as we are implementing in causal attention
        mask_bool= self.mask.bool()[:num_tokens, :num_tokens]
        masked_attention_weights= attention_weights.masked_fill_(mask_bool, -float('inf'))

        #---Normalize the masked attention weights
        attention_scores= torch.softmax(masked_attention_weights / keys.shape[-1] ** 0/5, dim= -1)

        #---apply dropout
        attention_scores= self.drop_out(attention_scores)

        #---Finding context vector
        """
        Shape of attention weights= (num_examples,num_heads,num_tokens,num_tokens)
        Shape of value= (b, num_tokens, num_heads, head_dim)
        """
        context_vector= (attention_scores @ values).transpose(1,2)

        #---Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vector = context_vector.contiguous().view(num_examples, num_tokens, self.d_out)
        context_vector = self.out_proj(context_vector) # optional projection

        return context_vector