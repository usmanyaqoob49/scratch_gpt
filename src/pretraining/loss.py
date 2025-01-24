"""
Module that will compute loss so that we can do evaluation and pretrain according to reducing that loss.
"""
import torch
def compute_loss(model, inputs, targets):
    model.eval()
    with torch.no_grad():
        logits= model(inputs)
    probabilities= torch.softmax(logits,
                                 dim= -1)
    output_token_ids= torch.argmax(probabilities, 
                                   dim= -1,
                                   keepdim= True)
    #taking out target probabilities so that we can maximise them to increase accuracy
    