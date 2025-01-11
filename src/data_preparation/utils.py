import os
import re


#To read the verdict dataset
def read_verdict(path):
    with open(path, "r", encoding= "utf-8") as f:
        raw_text= f.read()
    return raw_text

#To convert text to tokens
def convert_to_tokens(text):
    tokens= re.split(r'([, .]|\s)', text)
    result= [item for item in tokens if item.strip()] #remove white space
    return result