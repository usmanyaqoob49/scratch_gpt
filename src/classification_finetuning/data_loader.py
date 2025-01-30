from .data_prep import ClassDataset
from torch.utils.data import dataloader
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

    )