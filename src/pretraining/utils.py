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
    flat= tokens_ids.squeeze(0)
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

#Function to compute the average loss of complete data loader (all batches)
def calculate_loader_loss(data_loader, model, device, num_batches= None):
    total_loss= 0
    if len(data_loader)==0:
        return float('nan')
    elif num_batches is None:
        num_batches= len(data_loader)
    else:
        num_batches= min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i<num_batches:
            batch_loss= calculate_batch_loss(
                input_batch= input_batch,
                target_batch= target_batch,
                model= model,
                device= device
            )
            total_loss+= batch_loss.item()
        else:
            break
    return total_loss / num_batches #average loss over all the batches


#Function to evaluate model, will take model and data and will call loss functions
def evaluate_model(model, train_loader, validation_loader, 
                   device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loader_loss= calculate_loader_loss(data_loader= train_loader,
                                                 model= model,
                                                 device= device,
                                                 num_batches= eval_iter)
        validation_loader_loss= calculate_loader_loss(data_loader= validation_loader,
                                                      model= model,
                                                      device= device,
                                                      num_batches=  iter)
    model.train()
    return train_loader_loss, validation_loader_loss

#Function to print the text generate by model, this function will be used in while checking the results in training to make sure model is improving
def generate_print_sample_text(
        model,
        tokenizer,
        device, 
        start_context
):
    model.eval()