"""
This moduie has all the helper functions. 
Like:
    - Functiont to convert text input to tokens so that we can pass it to gpt module.
    - Function to convert tokens ids that we get from gpt to text.
"""
from src.gpt.utils import generate_text
import torch

#Function to convert text to tokens 
def text_to_tokens(tokenizer, text):
    encoded= tokenizer.encode(text, allowed_special= {'<|endoftext|>'})
    encoded_tensor= torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

#Function to convert tokens to text
def tokens_to_text(tokenizer, tokens_ids):
    flat= tokens_ids.sequeeze(0)
    text_tokens= tokenizer.decode(flat.tolist())
    return text_tokens

#Function to calculate the loss of batch
def calculate_batch_loss(input_batch, target_batch, model, device):
    input_batch= input_batch.to(device)
    target_batch= target_batch.to(device)
    logits= model(input_batch)
    batch_loss= torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )
    return batch_loss