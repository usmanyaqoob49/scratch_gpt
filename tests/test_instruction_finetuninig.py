import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.intruction_finetuning.data_loader import create_istructions_data_loader
from src.data_preparation.utils import gpt_tokenizer
from src.intruction_finetuning.utils import train_val_test_split
from src.classification_finetuning.utils import get_gpt_2_openai
from src.pretraining.utils import text_to_tokens, tokens_to_text, generate_text
import json

#----------Testing Data Loader
with open('./data/raw/instructions_data/instruction-data.json', 'r', encoding= 'utf-8') as file:
    data= json.load(file)
print('First data sample from the instructions dataset: ', data[0])
batch = [
    data[0],
    data[1],
]
gpt_2_tokenizer= gpt_tokenizer()
train_set, validation_set, test_set= train_val_test_split(json_data= data)
training_loader, validation_loader, test_loader= create_istructions_data_loader(training_dataset=train_set,
                                                                                validation_dataset= validation_set,
                                                                                test_dataset= test_set,
                                                                                tokenizer= gpt_2_tokenizer,
                                                                                batch_size= 8,
                                                                                num_workers= 0)

#------Using the function that initialize the gpt-architecture and then loads openai weights in it
openai_gpt_2_model= get_gpt_2_openai()
openai_gpt_2_model.eval()

# #------Testing not finetuned model on the instruction data
for input, target in validation_loader:
    print('First Input in first batch :', input[0])
    print('First target in first batch :', target[0])
    token_ids= generate_text(
        gpt_model= openai_gpt_2_model,
        idx= validation_loader[0],

    )
    break
