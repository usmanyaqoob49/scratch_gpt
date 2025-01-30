"""
This module has a function that process the dataset dataframe and then make a dataloader of the data.
"""
from .data_prep import ClassDataset
from torch.utils.data import DataLoader
from src.data_preparation.utils import gpt_tokenizer

def create_data_loaders(
        dataset,
        text_col_name,
        class_col_name,
        batch_size,
        num_workers,
        shuffle,
        drop_last
):
    tokenizer= gpt_tokenizer()
    processed_data= ClassDataset(
        data_df= dataset,
        text_col_name= text_col_name,
        class_col_name= class_col_name,
        tokenizer= tokenizer
    )
    data_loader= DataLoader(
        dataset= processed_data,
        batch_size= batch_size,
        shuffle= shuffle,
        num_workers= num_workers,
        drop_last= drop_last
    )
    return data_loader