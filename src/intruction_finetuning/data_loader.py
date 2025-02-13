""""
This Module has data loader function that takes datasets, prepare them, and create loaders from them.
"""
from .utils import InstructionDataset, custom_collate
from torch.utils.data import DataLoader
import torch
from functools import partial

def create_istructions_data_loader(training_dataset, validation_dataset, test_dataset, tokenizer, batch_size, num_workers):
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    customized_collate_function= partial(
        custom_collate,
        device= device,
        allowed_max_length= 1024
    )
    training_dataset= InstructionDataset(data= training_dataset, tokenizer= tokenizer)
    train_loader= DataLoader(
        dataset= training_dataset,
        batch_size= batch_size,
        collate_fn= customized_collate_function,
        shuffle= True,
        drop_last= True,
        num_workers= num_workers
    ) 
    validation_dataset= InstructionDataset(data= validation_dataset, tokenizer= tokenizer)
    validation_loader= DataLoader(
        dataset= validation_dataset,
        batch_size= batch_size,
        collate_fn= customized_collate_function,
        shuffle= False,
        drop_last= False,
        num_workers= num_workers
    )
    test_dataset= InstructionDataset(data= test_dataset, tokenizer= tokenizer)
    test_loader= DataLoader(
        dataset= test_dataset,
        batch_size= batch_size,
        collate_fn= customized_collate_function,
        shuffle= False,
        drop_last= False,
        num_workers= num_workers
    )
    return train_loader, validation_loader, test_loader
