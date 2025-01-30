from .data_prep import ClassDataset
from torch.utils.data import dataloader

def create_data_loaders(
        dataset,
        text_col_name,
        class_col_name,
        batch_size,
        num_workers,
        shuffle,
        drop_last
):
    processed_data= ClassDataset(
        data_df= dataset,
        text_col_name= text_col_name,
        class_col_name= class_col_name,
        
    )