"""Module that has a function that will get gpt-2 architecture, will load weights, get dataset, and then finetune the model."""
from .utils import get_gpt_2_openai
from .finetune import finetune_model
from .data_loader import create_data_loaders

def train(classification_dataset, text_col_name, labels_col_name, batch_size):
    training_loader, validation_loader= create_data_loaders(dataset= classification_dataset,
                                                            text_col_name= text_col_name,
                                                            class_col_name= labels_col_name,
                                                            batch_size= batch_size,
                                                            )
    gpt_2_openai= get_gpt_2_openai()
    