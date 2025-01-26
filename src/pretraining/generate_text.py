"""
Function for generating text with sampling techniques, like top-k sampling, temperature scaling etc.

Simple text generation function is implemented in src.gpt.utils
"""
import torch
from .utils import temperature_scaling, top_k_sampling

def generate_diverse(model, idx, max_new_tokens,
                     context_size, temperature= 0.0, top_k= None, eos_id= None):
    for _ in range(max_new_tokens):
        idx_context= idx[:, -context_size:]
        with torch.no_grad():
            