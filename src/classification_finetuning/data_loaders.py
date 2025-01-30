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
        self.label_col_name= labels_col_name
        self.dataset= pd.read_csv(csv_file)
        self.encoded_texts= [
            tokenizer.encode(text_row) for text_row in self.dataset[text_col_name]
        ]
        if max_length in None:
            self.max_length= self._max_length_encode()
            self.encoded_texts= [encoded_text[:self.max_length] for encoded_text in self.encoded_texts]
        self.encoded_texts= [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]
    def __getitem__(self, index):
        encoded= self.encoded_texts[index]
        label= self.dataset.iloc[index][self.label_col_name]
        return (
            torch.tensor(encoded, dtype= torch.long),
            torch.tensor(label, dtype= torch.long)
        )
    def __len__(self):
        return len(self.dataset)
    def _max_length_encode(self):
        max_length= 0
        for encoded_text in self.encoded_texts:
            if len(encoded_text)>max_length:
                max_length= len(encoded_text)
        return max_length
