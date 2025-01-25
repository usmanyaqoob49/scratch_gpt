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

#Function to calculate the loss of single batch
def calculate_batch_loss(input_batch, target_batch, model, device):
    input_batch= input_batch.to(device)
    target_batch= target_batch.to(device)
    logits= model(input_batch)
    batch_loss= torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )
    return batch_loss

#Function to compute the loss of complete data loader (all batches)
def calculate_loader_loss(data_loader, model, device, num_batches= None):
    total_loss= 0
    if len(data_loader)==0:
        return float('nan')
    elif num_batches is None:
        num_batches= len(data_loader)
    else:
        num_batches= min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        