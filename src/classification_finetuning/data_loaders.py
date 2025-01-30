"""
This module has class that created data loader of csv dataset by converting text to encoded text
and then padding the text to make them of equal lenght.
"""
import torch
from torch.utils.data import Dataset

class classDataset(Dataset):
    