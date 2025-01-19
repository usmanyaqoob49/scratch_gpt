from src.data_preparation.utils import gpt_tokeinzer
from src.gpt.gpt_model import GPTModel
from src.gpt.utils import GPT_CONFIG_124M
import torch

torch.manual_seed(123)
gpt= GPTModel(cfg= GPT_CONFIG_124M)

batch= []
text1= 'Every effort moves you'
text2= 'Every day holds a'

batch.append(torch.tensor(gpt_tokeinzer.encode(text1)))
batch.append(torch.tensor(gpt_tokeinzer.encode(text2)))
