"""
This module will  have function to load the gpt-2 open ai weights using helper function from gpt_downlaod.
"""
import torch
import tensorflow as tf

def load_gpt_openai_weights(weights_dir):
    tf_checkpoints= tf.train.latest_checkpoint()