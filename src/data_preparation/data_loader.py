"""
This module has data loader function.

Data loader is something that convert data to batches, shuffle it, prepare it and return input and output 
in the form of bathces.
"""
import torch
from .utils import gpt_tokeinzer
from .data_sampling import GPTDatasetV1

