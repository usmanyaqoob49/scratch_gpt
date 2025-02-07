"""
This module has the function to finetune the gpt-2 model on the classificatio dataset.

Evaluation Frequency decides after how many batches that will be processed, evaluation will happen.

"""
import torch
torch.manual_seed(123)
from .utils import batch_classification_loss, evaluate_model, loader_classification_accuracy

def finetune_model(model, train_loader, validation_loader, optimizer, device, num_epochs, eval_frequency, eval_iter):
    examples_seen= 0
    global_step= -1
    training_loss, validation_loss= [], []
    training_accuray, validation_accuracy= [], []

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            batch_loss= batch_classification_loss(gpt_model= model,
                                                  input_batch= input_batch,
                                                  target_batch= target_batch,
                                                  device= device)
            batch_loss.backward()
            optimizer.step()
            examples_seen+= input_batch.shape[0]
            global_step+=1  
            if global_step % eval_frequency == 0:
                train_loss, val_loss= evaluate_model(model= model,
                                                     train_loader= train_loader,
                                                     val_loader= validation_loader,
                                                     device= device,
                                                     eval_iter= eval_iter)
                print(f"Epoch: {epoch+1} and Global Step: {global_step} \n Training Loss: {train_loss} and Validation Loss: {val_loss}")
        training_loader_accuracy= loader_classification_accuracy(data_loader= train_loader,
                                                                 model= model,
                                                                 device= device,
                                                                 num_batches= eval_iter)
        validation_loader_accuracy= loader_classification_accuracy(data_loader= validation_loader,
                                                                   model= model,
                                                                   device= device,
                                                                   num_batches= eval_iter)
        training_accuray.append(training_loader_accuracy)
        validation_accuracy.append(validation_loader_accuracy)
        print(f"Training Accuracy after epoch: {epoch} is {training_loader_accuracy}")
        print(f"Validation Accuracy after epoch: {epoch}")



    

