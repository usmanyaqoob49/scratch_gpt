import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preparation.utils import gpt_tokeinzer
from src.gpt.gpt_model import GPTModel
from src.gpt.utils import GPT_CONFIG_124M
from src.gpt.layer_norm import LayerNorm
from src.gpt.utils import FeedForward
from src.transformer.transformer_block import Transformer
from src.gpt.utils import generate_text
import torch

#---------Testing Layer Norm class
batch_examples= torch.randn(2,5)
norm= LayerNorm(emb_dims= 5)
out_norm= norm.forward(batch_examples)
print("Normalized Output: ", out_norm)
print("Mean of Normalized output: ",  out_norm.mean(dim= -1, keepdim= True))
print("Varaince of the normalized output: ", out_norm.var(dim= -1, keepdim= True, unbiased= False))
print('------------------------------------------------------------------\n\n')
#---------Testing Layer feed forward
ffn= FeedForward(cfg= GPT_CONFIG_124M)
x= torch.rand(2, 3, 768)
out= ffn.forward(x)
print('Output shape of feed forward: ', out.shape)
print('------------------------------------------------------------------\n\n')

##---------Testing Transformer Block
tb= Transformer(cfg= GPT_CONFIG_124M)
x= torch.randn(2, 4, 768)
print("Input of transfomer shape: ", x.shape)
output= tb.forward(x=x)
print("Transformer Block Output Shape: ", output.shape)
print('------------------------------------------------------------------\n\n')


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
print("\n\nLogits shape: ", logits.shape)

total_parameters= sum(p.numel() for p in gpt.parameters())
print("Total parameters in GPT Model:", total_parameters)
print('------------------------------------------------------------------\n\n')


#---------Testing generate text function that uses the gpt model to generate next token
context= "Hello I am"
encoded= tokenizer.encode(text= context)
print("Encoded Text of Context: ", encoded)
encoded_tensor= torch.tensor(encoded).unsqueeze(0)  #to add batch dimension
print("Encoded tensor of Context shape: ", encoded_tensor.shape)

gpt.eval() #putting in evaluation mode as we are going to predict
output= generate_text(gpt_model= gpt,
                      idx= encoded_tensor,
                      max_new_tokens= 6,
                      context_size= GPT_CONFIG_124M['context_length'])
print("Output from GPT model: ", output)

decoded_text= tokenizer.decode(output.squeeze(0).to_list())


print('------------------------------------------------------------------\n\n')
