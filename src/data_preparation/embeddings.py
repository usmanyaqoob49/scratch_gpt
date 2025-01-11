"""
This file has module that creates embeddiings from token ids.
"""
import torch

def create_embeddings(input_ids, vocab_size, output_dim):
    torch.manual_seed(123)
    embedding_layer= torch.nn.Embedding(vocab_size, output_dim)
    input_embeddings= embedding_layer(input_ids)
