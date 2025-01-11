"""
This module has data loader function.

Data loader is something that convert data to batches, shuffle it, prepare it and return input and output 
in the form of bathces.
"""
import torch
from .utils import gpt_tokeinzer
from .data_sampling import GPTDatasetV1
from torch.utils.data import Dataset, DataLoader

def create_data_loader_v1(
        txt,
        batch_size= 4,
        max_length= 256,
        stride= 128,
        shuffle= True,
        drop_last= True,
        num_workers= 0
):
    tokenizer= gpt_tokeinzer()
    dataset= GPTDatasetV1(text= txt,
                          tokenizer= tokenizer, 
                          max_length= max_length,
                          stride= stride)
    dataloader= DataLoader(
        dataset= dataset,
        batch_size= batch_size,
        shuffle= shuffle,
        drop_last= drop_last,
        num_workers= num_workers
    )
    return dataloader
