"""Module that has a function that will get gpt-2 architecture, will load weights, get data loaders, and then finetune the model."""
from .utils import get_gpt_2_openai
from .finetune import finetune_model
import torch

def train(training_loader, validation_loader, num_epochs, eval_frequency, batch_size):
    gpt_2_openai= get_gpt_2_openai()
    optimizer= torch.optim.AdamW(params= gpt_2_openai.parameters(),
                                 lr= 5e-5,
                                 weight_decay= 0.1)
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    finetuned_model, training_accuray, validation_accuracy, training_loss, validation_loss, examples_seen= finetune_model(model= gpt_2_openai,
                                                                                                         train_loader= training_loader,
                                                                                                         validation_loader= validation_loader,
                                                                                                         optimizer= optimizer,
                                                                                                         device= device,
                                                                                                         num_epochs= num_epochs,
                                                                                                         eval_frequency= eval_frequency,
                                                                                                         eval_iter= batch_size)
    
    return finetuned_model, training_accuray, validation_accuracy, training_loss, validation_loss, examples_seen