import os
import re


#To read the verdict dataset
def read_verdict(path):
    with open(path, "r", encoding= "utf-8") as f:
        raw_text= f.read()
    return raw_text
