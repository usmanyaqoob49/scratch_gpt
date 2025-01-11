"""
This module contains a SimpleTokenizer class which is used to encode and decode text using a given vocabulary.
"""
from .utils import convert_to_tokens, read_verdict, vocab_assign_token_id, tokenid_to_token
import re

class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int= vocab
        self.int_to_str= tokenid_to_token(vocab= vocab)

    def encode(self, text):
        tokens= convert_to_tokens(text= text)
        #adding extra special tokens--->if token id for specific token is not present in the vocabulary just assign it <|unk|>
        processed= [
            item if item in self.str_to_int
            else "<|unk|>" for item in tokens
        ]
        ids= [self.str_to_int[s] for s in processed]
        return ids
    
    def decode(self, ids):
        text= " ".join([self.int_to_str[i] for i in ids])
        proper_text= re.sub(r'\s+([,.:;?!"()\'])', r'\1', text) #Replace spaces before the specified punctuations
        return proper_text