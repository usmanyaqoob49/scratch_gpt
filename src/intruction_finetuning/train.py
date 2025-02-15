"""Module that will load the gpt-2 model and finetune fucntion and it will perform instruction finetuning using the prepared data."""
import torch
from src.classification_finetuning.finetune import finetune_model
from src.classification_finetuning.utils import get_gpt_2_openai, gpt_2_124m_configurations

gpt_2_openai= get_gpt_2_openai()
