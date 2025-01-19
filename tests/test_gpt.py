import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preparation.utils import gpt_tokeinzer
from src.gpt.gpt_model import GPTModel
from src.gpt.utils import GPT_CONFIG_124M
from src.gpt.layer_norm import LayerNorm
import torch

#---------Testing GPTModel class
tokenizer= gpt_tokeinzer()
torch.manual_seed(123)
batch= []
text1= 'Every effort moves you'
text2= 'Every day holds a'

batch.append(torch.tensor(tokenizer.encode(text1)))
batch.append(torch.tensor(tokenizer.encode(text2)))

batch = torch.stack(batch, dim=0)   #to make batch row by row
print("Batch of examples that we are using: ", batch)

gpt= GPTModel(cfg= GPT_CONFIG_124M)
logits= gpt.forward(in_idx= batch)
print("Resulting Logits: ", logits)
print("Logits shape: ", logits.shape)


#---------Testing Layer Norm class
norm= LayerNorm(emb_dims= 5)
out_norm= LayerNorm.forward(batch)
print("Normalized Output: ", out_norm)
print("Mean of Normalized output: ",  out_norm.mean)
print("Varaince of the normalized output: ", out_norm.variance)