"""Module that has a function that will get gpt-2 architecture, will load weights, get dataset, and then finetune the model."""
from .utils import get_gpt_2_openai
from .finetune import finetune_model
from .data_loader import create_data_loaders

def train(training_loader, validation_loader):
    gpt_2_openai= get_gpt_2_openai()
    