"""
Function for generating text with sampling techniques, like probablistic sampling, top-k sampling, temperature scaling etc instead
of always selecting max probablity token (Greedy Sampling).

Simple text generation function is implemented in src.gpt.utils
"""
import torch
from .utils import temperature_scaling, top_k_sampling

def generate_diverse(model, idx, max_new_tokens,
                     context_size, temperature= 0.0, top_k= None, eos_id= None):
    for _ in range(max_new_tokens):
        idx_context= idx[:, -context_size:]
        with torch.no_grad():
            logits= model(idx_context)
        new_tokens_logits= logits[:, -1, :]

        if top_k is not None:
            new_tokens_logits= top_k_sampling(logits= new_tokens_logits, 
                                         k= top_k)
        if temperature != 0.0:
            new_tokens_logits= temperature_scaling(logits= new_tokens_logits,
                                               temperature= temperature)
            probabilites= torch.softmax(new_tokens_logits, dim= -1)
            idx_next= torch.multinomial(input= probabilites, num_samples= 1)
        else:
            idx_next= torch.argmax(logits, dim= -1, keepdim= True)
        if idx_next == eos_id:
            break
        idx= torch.cat((idx, idx_next))
    return idx
            