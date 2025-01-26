"""
This module will  have function to load the gpt-2 open ai weights using helper function from gpt_downlaod.
"""
import torch
import tensorflow as tf
from .gpt_download import load_gpt2_params_from_tf_ckpt
def load_gpt_openai_weights(weights_dir):
    tf_checkpoints= tf.train.latest_checkpoint(checkpoint_dir= weights_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params