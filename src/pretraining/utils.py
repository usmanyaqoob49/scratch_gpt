"""
This moduie has all the helper functions. 
Like:
    - Functiont to convert text input to tokens so that we can pass it to gpt module.
    - Function to convert tokens ids that we get from gpt to text.
"""
from src.gpt.utils import generate_text, GPT_CONFIG_124M
from src.data_preparation.data_loader import create_data_loader_v1
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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

#Function to make train and test loaders of any dataset
def make_train_validation_loader(text_data,
                                train_data_ratio):
    split_index= int(train_data_ratio * len(text_data))
    train_data_text= text_data[:split_index]
    validation_data_text= text_data[split_index:]

    train_loader= create_data_loader_v1(
        txt= train_data_text,
        batch_size= 2,
        max_length= GPT_CONFIG_124M['context_length'],
        stride= GPT_CONFIG_124M['context_length'],
        drop_last= True,
        shuffle= True,
        num_workers= 0
    )

    validation_loader= create_data_loader_v1(
        txt= validation_data_text,
        batch_size= 2,
        max_length= GPT_CONFIG_124M['context_length'],
        stride= GPT_CONFIG_124M['context_length'],
        drop_last= False,
        shuffle= False,
        num_workers= 0
    )
    return train_loader, validation_loader

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
    total_loss= 0.0
    if len(data_loader)== 0:
        return float('nan')
    elif num_batches is None:
        num_batches= len(data_loader)
    else:
        num_batches= min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i<num_batches:
            loss= calculate_batch_loss(input_batch, target_batch, model, device)
            total_loss+= loss.item()
        else:
            break
    return total_loss/ num_batches


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
                                                      num_batches=  eval_iter)
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
    context_size= model.pso_emb.weight.shape[0]
    encoded= text_to_tokens(tokenizer= tokenizer,
                           text= start_context).to(device)
    
    with torch.no_grad():
        token_ids= generate_text(
            gpt_model= model,
            idx= encoded,
            max_new_tokens= 50,
            context_size= context_size
        )
    decoded_text= tokens_to_text(tokenizer= tokenizer,
                                 tokens_ids= token_ids)
    print(decoded_text.replace("\n", " "))  
    model.train()

#Function to plot the training and validation loss-->will return
def plot_losses(training_loss, validation_loss, epochs_seen, tokens_seen):
    fig, ax1= plt.subplot(fig_size= (5,3))

    ax1.plot(epochs_seen, training_loss, label= 'Training Loss')
    ax1.plot(epochs_seen, validation_loss, label= 'Validation Loss', linestyle= '-.')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(loc= 'upper right')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    ax2= ax1.twiny()
    ax2.plot(tokens_seen, training_loss, alpha= 0)
    ax2.set_xlabel('Tokens Seen')

    fig.tight_layout()
    path= ".src/pretraining/plots/loss-plot.pdf"
    plt.savefig(path)
    return path

#Temperature scaling function--->That we can use for selection of logits
def temperature_scaling(logits, temperature):
    return logits / temperature

#Function for selection of top-k logits for Top-k Selection--->Selects top-k most probable logits to make sure we are using high accurate tokens when we apply randomness
