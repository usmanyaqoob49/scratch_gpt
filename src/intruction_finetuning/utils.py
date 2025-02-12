"""
Helping functions for the Instruction Finetuning of the LLM model.
"""
import torch

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
