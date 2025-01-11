"""
As LLM gnerates next word using previous words as context, so we have to create dataset in the way that 
we have input and output/target shifted one to right.

input: Hi I am.
output: I am Usman.

Also it has GPTDataset class that will convert the dataset to dataset required for GPT training.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from .utils import gpt_tokeinzer

class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids= []
        self.output_ids= []

        token_ids= tokenizer.encode(text, allowed_special= {"<|endoftext|>"})
        
        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids)-max_length, stride):
            input_chunk= token_ids[i:i+max_length]  #ipnut will be start to max_length
            output_chunk= token_ids[i+1:i+max_length+1] #output will be one word to right 

            self.input_ids.append(torch.tensor(input_chunk))
            self.output_ids.append(torch.tensor(output_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.output_ids[index]
    
