import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_preparation.tokenizer import SimpleTokenizer
from src.data_preparation.utils import read_verdict, convert_to_tokens, vocab_assign_token_id, gpt_tokeinzer
from src.data_preparation.data_sampling import GPTDatasetV1
from src.data_preparation.data_loader import create_data_loader_v1

verdict_text= read_verdict(path= "./data/raw/the-verdict.txt")
print('verdict_text_lenght= ', len(verdict_text))

verdict_text_tokens= convert_to_tokens(text= verdict_text)
print('Number of tokens after converting text to tokens: ', len(verdict_text_tokens))

verdict_text_vocabulary= vocab_assign_token_id(tokens= verdict_text_tokens)

tokenizer= SimpleTokenizer(vocab= verdict_text_vocabulary)
sample_text= """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids= tokenizer.encode(text= sample_text)
print('Token ids assigned by encoder of tokenizer: ', ids)

text_from_ids= tokenizer.decode(ids= ids)
print('Text assigned from decoder using token ids', text_from_ids)

tokenizer= gpt_tokeinzer()
dataset= GPTDatasetV1(text= verdict_text, 
                      tokenizer= tokenizer,
                      max_length= 4,
                      stride= 1)

print('input ids 1: ', dataset.input_ids[1])
print('output ids 1: ', dataset.output_ids[1])
