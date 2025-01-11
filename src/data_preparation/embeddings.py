"""
This file has module that creates embeddiings from token ids.

It has implementation of tokens embeddings + positional embeddings too.
"""
import torch

def create_embeddings(input_ids, vocab_size, context_length, output_dim):
    torch.manual_seed(123)
    embedding_layer= torch.nn.Embedding(vocab_size, output_dim)
    input_embeddings= embedding_layer(input_ids)

    positional_embeddings_layer= torch.nn.Embedding(context_length, output_dim)
    positional_embeddings_layer= positional_embeddings_layer(torch.arange(context_length)) #it will have numbers from 0 to input_lenght-1

    return input_embeddings
