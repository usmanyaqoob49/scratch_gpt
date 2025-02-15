"""Module that will load the gpt-2 model and finetune fucntion and it will perform instruction finetuning using the prepared data. """
import torch
from src.classification_finetuning.finetune import finetune_model
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