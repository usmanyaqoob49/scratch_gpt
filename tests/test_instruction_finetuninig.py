from src.intruction_finetuning.utils import InstructionDataset, custom_collate
from src.data_preparation.utils import gpt_tokenizer

inputs_1 = "Hi how are you doing?"
inputs_2 = "I am doing great"
batch = [
    inputs_1,
    inputs_2,
]
gpt_2_tokenizer= gpt_tokenizer()
instruction_dataset= InstructionDataset(data= batch, tokenizer= gpt_2_tokenizer)
print("Encoded: ", instruction_dataset.encoded_text)


