"""
File has all the function used in preprocessing.
"""
import os
import re
import tiktoken

#To read the verdict dataset
def read_verdict(path):
    with open(path, "r", encoding= "utf-8") as f:
        raw_text= f.read()
    return raw_text

#To convert text to tokens
def convert_to_tokens(text):
    pattern= r'([,.:;?_!"()\']|--|\s)'  #pattern for handling punctuations and splitting every string
    tokens= re.split(pattern, text)
    result= [item for item in tokens if item.strip()] #remove white space
    return result

#assign token id-->making vocabulary to assign token id
def vocab_assign_token_id(tokens):
    all_words= sorted(set(tokens))
    vocab= {token:token_id for token_id,token in enumerate(tokens)}
    return vocab

#to convert token_ids to tokens text
def tokenid_to_token(vocab):
    return {token_id:token for token, token_id in vocab.items()}

#we will use gpt tokenizer by using library called tiktoken
def gpt_tokeinzer():
    tokenizer= tiktoken.get_encoding('gpt-2')