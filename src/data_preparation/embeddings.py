"""
This file has module that creates embeddiings from token ids.
"""
import torch

def create_embeddings(input_ids, vocab_size, output_dim):
    