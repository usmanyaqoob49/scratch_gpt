"""
As LLM gnerates next word using previous words as context, so we have to create dataset in the way that 
we have input and output/target shifted one to right.

Also it has GPTDataset class that will convert the dataset to dataset required for GPT training.
"""
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    