"""
This module has function to train the LLM on the given data loaders. We will use it for the instruction finetuning.
"""
import torch
from .utils import calculate_batch_loss, evaluate_model, generate_print_sample_text

def train_model(model, train_loader, validation_loader,
                optimizer, device, num_epochs,
                eval_freq, eval_iter, start_context,
                tokenizer):
    training_loss, validation_loss, track_tokens_seen= [], [], []
    tokens_seen, global_step= 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            batch_loss= calculate_batch_loss(input_batch= input_batch,
                                             target_batch= target_batch,
                                             model= model,
                                             device= device)
            batch_loss.backward()
            optimizer.step()
            tokens_seen+= input_batch.numel()
            global_step+=1

            if global_step%eval_freq == 0:
                train_loss, val_loss= evaluate_model(
                    model= model,
                    train_loader= train_loader,
                    validation_loader= validation_loader,
                    device= device,
                    eval_iter= eval_iter
                )
                training_loss.append(train_loss)
                validation_loss.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                
        generate_print_sample_text(model= model,
                                   tokenizer= tokenizer,
                                   device= device,
                                   start_context= start_context)
    return model, training_loss, validation_loss, track_tokens_seen
        


import torch
from src.pretraining.train import train_model
from src.classification_finetuning.utils import get_gpt_2_openai, gpt_2_124m_configurations

def instruction_finetune_gpt2(train_loader, validation_loader, optimizer, num_epochs, evaluation_frequency, batch_size):
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpt_2_openai= get_gpt_2_openai()
    intruction_finetuned_model, training_accuray, validation_accuracy, training_loss, validation_loss, examples_seen= finetune_model(
        model= gpt_2_openai,
        train_loader= train_loader,
        validation_loader= validation_loader,
        optimizer= optimizer,
        device= device,
        num_epochs= num_epochs,
        eval_frequency= evaluation_frequency, 
        eval_iter= batch_size
    )
    return intruction_finetuned_model, training_accuray, validation_accuracy, training_loss, validation_loss, examples_seen