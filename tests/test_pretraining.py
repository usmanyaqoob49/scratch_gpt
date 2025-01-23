"""
Test cases for different modules of pretraining.
"""
from src.pretraining.utils import text_to_tokens, tokens_to_text
from src.gpt.gpt_model import GPTModel
from src.gpt.utils import GPT_CONFIG_124M
from src.data_preparation.utils import gpt_tokeinzer
from src.gpt.utils import generate_text

tokenizer= gpt_tokeinzer()
gpt_model= GPTModel(cfg= GPT_CONFIG_124M)

start_context = "Every effort moves you"
token_ids_result= generate_text(
    gpt_model= gpt_model,
    idx= text_to_tokens(tokenizer= tokenizer,
                        text= start_context),
    max_new_tokens= 10,
    context_size= GPT_CONFIG_124M['context_length'],
)

print("Testing text_to_tokens Function:")