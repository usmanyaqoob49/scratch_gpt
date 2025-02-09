"""Module that has a function that will get gpt-2 architecture, will load weights, get dataset, and then finetune the model."""
from .utils import get_gpt_2_openai
from .finetune import finetune_model

def train(training_loader, validation_loader, optimizer, device, num_epochs, eval_frequency, batch_size):
    gpt_2_openai= get_gpt_2_openai()
    training_accuray, validation_accuracy, training_loss, validation_loss, examples_seen= finetune_model(model= gpt_2_openai,
                                                                                                         train_loader= training_loader,
                                                                                                         validation_loader= validation_loader,
                                                                                                         optimizer= optimizer,
                                                                                                         device= device,
                                                                                                         num_epochs= num_epochs,
                                                                                                         eval_frequency= eval_frequency,
                                                                                                         eval_iter= batch_size)
    