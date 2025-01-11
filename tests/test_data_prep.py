from src.data_preparation.tokenizer import SimpleTokenizer
from src.data_preparation.utils import read_verdict, convert_to_tokens, vocab_assign_token_id

verdict_text= read_verdict(path= "")
tokenizer= SimpleTokenizer()