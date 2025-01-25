import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_preparation.tokenizer import SimpleTokenizer
from src.data_preparation.utils import read_verdict, convert_to_tokens, vocab_assign_token_id, gpt_tokenizer, tokenid_to_token
from src.data_preparation.data_sampling import GPTDatasetV1
from src.data_preparation.data_loader import create_data_loader_v1
from src.data_preparation.embeddings import create_embeddings

#reading dataset (raw)
verdict_text= read_verdict(path= "./data/raw/the-verdict.txt")
print('verdict_text_lenght= ', len(verdict_text))

#converting to tokens
verdict_text_tokens= convert_to_tokens(text= verdict_text)
print('Number of tokens after converting text to tokens: ', len(verdict_text_tokens))

#assigning token ids
verdict_text_vocabulary= vocab_assign_token_id(tokens= verdict_text_tokens)

tokenizer= SimpleTokenizer(vocab= verdict_text_vocabulary)
sample_text= """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids= tokenizer.encode(text= sample_text)
print('Token ids assigned by encoder of tokenizer: ', ids)

text_from_ids= tokenizer.decode(ids= ids)
print('Text assigned from decoder using token ids', text_from_ids)

tokenizer= gpt_tokenizer()

#testing dataset class that converts simple tokens to proper input and output tokens 
dataset= GPTDatasetV1(text= verdict_text, 
                      tokenizer= tokenizer,
                      max_length= 4,
                      stride= 1)

print('input ids 1: ', dataset.input_ids[1])
print('output ids 1: ', dataset.output_ids[1])

#testing dataloader that takes the dataset of input and output and convert them to batches
dataloader= create_data_loader_v1(
    txt= verdict_text,
    batch_size= 4,
    max_length= 4,
    stride=1,
    shuffle= False
)
print('Result by dataloader when we gave it dataset: (First 5 input/outputs)')
for i, (input, output) in enumerate(dataloader):
    if i<5:
        print(f'input Batch {i}: ', input)
        print(f'output Batch {i}: ', output)

#testing embeddings function
data= iter(dataloader)
input, output= next(data)
print('shape of input: ', input.shape)
embedded_input= create_embeddings(input_ids= input, vocab_size= 50257, context_length= 4, output_dim= 256)
