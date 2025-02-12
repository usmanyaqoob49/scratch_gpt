"""
Helping functions for the Instruction Finetuning of the LLM model.
"""
import torch
from torch.utils.data import Dataset

#Function to format the instructions data in a specific aplaca prompt template
def format_input(input_entry):
    instructions_text= (
        "Below is an instruction that describes the task. "
        "Write a response that appropriately completes the request. "
        f"\n\n ### Instruction: \n{input_entry['instruction']}"
    )
    input_text= (
        f"\n\n### Input: \n{input_entry['input']}" if input_entry["input"] else ""
    )
    response_text= (
        f"\n\n ###Response: \n{input_entry['output']}"
    )
    final_input_text= instructions_text + input_text + response_text
    return final_input_text

#class for preparing the datasets before being loaded in the data loader--->Format the input, tokenize it
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        super().__init__()
        self.encoded_text= []
        self.data= data
        for entry in data:
            aplaca_formatted_text= format_input(input_entry= entry)
            tokenized_text= tokenizer.encode(aplaca_formatted_text)
            self.encoded_text.append(tokenized_text)
    def __getitem__(self, index):
        return self.encoded_text(index)
    def __len__(self):
        return len(self.encoded_text)
    
#Function to perform-->We will get tokenized batches, we will pad them. We will replace padding number with -100 so it can be ignored during training, in the end we will make target by shifting input by 1
def custom_collate(batch, pad_token_id= 50256, ignore_index= -100, allowed_max_length= None, device= 'cpu'):
    max_length_in_batch= max(len(entry) + 1 for entry in batch)
    input_ids, target_ids= [], []
    for entry in batch:
        padded_entry= (
            entry + [pad_token_id] * (max_length_in_batch - (len(entry) + 1))
        )
        input_tensor= torch.tensor(padded_entry)
        target_tensor= torch.tensor(padded_entry[1:] + [pad_token_id])
        mask= target_tensor == pad_token_id
        padded_indices= torch.nonzero(mask).sequeeze() 
        if padded_indices.numel() > 1:
            target_tensor[padded_indices[1:]]= ignore_index #leave first padded_token so that model can know that text ended here
        if allowed_max_length is not None:
            allowed_input= input_tensor[:allowed_max_length]
            allowed_target= target_tensor[:allowed_max_length]
        input_ids.append(allowed_input)
        target_ids.append(allowed_target)
    input_tensors= torch.stack(input_ids).to(device)
    target_tensors= torch.stack(target_ids).to(device)
    return input_tensor, target_tensor


