"""
Test cases for different modules of pretraining.
"""
from src.pretraining.utility import text_to_tokens, tokens_to_text
from src.gpt.gpt_model import GPTModel
from src.gpt.utils import GPT_CONFIG_124M

gpt_model= GPTModel(cfg= GPT_CONFIG_124M)
print("Testing text_to_tokens Function:")