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

#class for preparing the datasets before being loaded in the data loader
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
    
