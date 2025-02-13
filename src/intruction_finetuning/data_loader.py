""""
This Module has data loader function that takes datasets, prepare them, and create loaders from them.
"""
from .utils import InstructionDataset
from torch.utils.data import DataLoader
import torch

def create_istructions_data_loader(training_dataset, validation_dataset, test_dataset, tokenizer, batch_size, num_workers):
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu') 