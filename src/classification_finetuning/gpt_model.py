"""
This module has function that will initialize the gpt architecture, will load openai downloaded weights for 
gpt-2, will load these weights in the gpt architecture and in the end it will return the gpt-2 model loaded with openai weights.
"""
import torch
from src.gpt.gpt_model import GPTModel
from src.pretraining.utils import load_gpt2_params_from_tf_ckpt
from src.pretraining.loading_weight_into_gpt import loads_weight_into_gpt
from src.pretraining.utils import gpt_2_124m_configurations

def get_gpt_2_openai():
    gpt_2_architecture= GPTModel(cfg= gpt_2_124m_configurations)
    openai_gpt2_weights= load_gpt2_params_from_tf_ckpt(ckpt_path= './models/gpt-2/124M/124M',
                                                       settings= )
