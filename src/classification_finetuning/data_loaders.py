"""
This module has class that created data loader of csv dataset by converting text to encoded text
and then padding the text to make them of equal lenght.
"""
import torch
from torch.utils.data import Dataset
import pandas as pd

class classDataset(Dataset):
    def __init__(self, csv_file, text_col_name, labels_col_name, 
                 tokenizer, max_length= None, pad_token_id= 50256):
        super().__init__()
        self.dataset= pd.read_csv(csv_file)
        self.encoded_text= [
            tokenizer.encode(text_row) for text_row in self.dataset[text_col_name]
        ]
        if max_length in None:
            self.max_length= self._max_length_encode()