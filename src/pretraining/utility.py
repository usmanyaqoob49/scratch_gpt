"""
This moduie has all the helper functions. 
Like:

"""
from src.gpt.utils import generate_text
import tiktoken
import torch

#Function to convert text to tokens 
def text_to_tokens(tokenizer, text):
    encoded= tokenizer.encode(text, allowed_special= {'<|endoftext|>'})
    encoded_tensor= torch.ten
    return encoded