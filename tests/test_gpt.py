from src.data_preparation.utils import gpt_tokeinzer
from src.gpt.gpt_model import GPTModel
from src.gpt.utils import GPT_CONFIG_124M
import torch

torch.manual_seed(123)
gpt= GPTModel(cfg= GPT_CONFIG_124M)
