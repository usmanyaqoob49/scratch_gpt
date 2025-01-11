"""
As LLM gnerates next word using previous words as context, so we have to create dataset in the way that 
we have input and output/target shifted one to right.
"""
import torch
from torch.utils.data import Dataset, DataLoader
