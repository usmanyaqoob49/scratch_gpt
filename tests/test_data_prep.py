import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_preparation.tokenizer import SimpleTokenizer
from src.data_preparation.utils import read_verdict, convert_to_tokens, vocab_assign_token_id

verdict_text= read_verdict(path= "./data/raw/the-verdict.txt")
print('verdict_text_lenght= ', len(verdict_text))

verdict_text_tokens= convert_to_tokens(text= verdict_text)
print('Number of tokens after converting text to tokens: ', len(verdict_text_tokens))

verdict_text_vocabulary= vocab_assign_token_id(tokens= verdict_text_tokens)
for key, item in verdict_text_vocabulary.items():
    print('First items pair in vocbulary: ')
    print(key, item)



tokenizer= SimpleTokenizer()