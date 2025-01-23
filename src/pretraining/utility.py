"""
This moduie has all the helper functions. 
Like:
    - Functiont to convert text input to tokens so that we can pass it to gpt module.

"""
from src.gpt.utils import generate_text
import tiktoken
import torch

#Function to convert text to tokens 
def text_to_tokens(tokenizer, text):
    encoded= tokenizer.encode(text, allowed_special= {'<|endoftext|>'})
    encoded_tensor= torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

#Function to convert tokens to text
def tokens_to_text(tokenizer, tokens_ids):
    flat= tokens_ids.sequeeze(0)
    