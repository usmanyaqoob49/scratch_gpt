import pandas as pd 
import torch

#Function that will balance the classification dataset based on minimum number of classes present in it (like all class frequence will be equal to lowest repeated class)
def balance_dataset(dataset_path, classes_column_name):
    data= pd.read_csv(dataset_path)
    data_count_df= pd.DataFrame(data[classes_column_name].value_counts())
    lowest_frequency= data_count_df['count'].min()
    balanced_data= data.groupby(classes_column_name).apply(lambda x: x.sample(n= lowest_frequency, random_state= 42)).reset_index(drop= True)
    return balanced_data

#Function to find the unique claasses and will return their mapping to a number so we can convert class to a number
def class_mapping(classes_list):
    unique_classes= sorted(set(classes_list))
    class_labels_mapping= {cls:number for number, cls in enumerate(unique_classes)}
    return class_labels_mapping

#Function to make the train, test and validation set
def train_val_test_split(data, train_frac= 0.7, validation_frac= 0.1):
    shuffled_data= data.sample(frac= 1, random_state= 42).reset_index(drop= True)
    train_split_index= int(len(shuffled_data) * train_frac)
    validation_split_index= train_split_index + int(len(shuffled_data) * validation_frac)
    train_set= shuffled_data[:train_split_index]
    validation_set= shuffled_data[train_split_index:validation_split_index]
    test_set= shuffled_data[validation_split_index:]
    return train_set, validation_set, test_set

#Function to map the lables of the classes to given numbers mapping
def map_classes(dataset, class_col_name, class_mapping_dict):
    dataset[class_col_name]= dataset[class_col_name].map(class_mapping_dict)
    return dataset

#Function to find the classification accuracy of the loader
def loader_classification_accuracy(data_loader, model, device, num_batches= None):
    model.eval()
    correct_predictions, num_examples= 0, 0
    if num_batches is None:
        num_batches= len(data_loader)
    else:
        num_batches= min(len(data_loader), num_batches)
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i<num_batches:
            input_batch, target_batch= input_batch.to(device), target_batch.to(device)
            with torch.no_grad():
                logits= model(
                    input_batch
                )
                last_token_logits= logits[:, -1, :]
            predicted_labels= torch.argmax(last_token_logits, 
                                           dim= -1)
            num_examples+= predicted_labels.shape[0]
            correct_predictions+= (predicted_labels==target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples

#Function to find the classification loss via cross entropy of a batch
def batch_classification_loss(gpt_model, input_batch, target_batch, device):
    gpt_model.eval()
    input_batch, target_batch= input_batch.to(device), target_batch.to(device)
    logits= gpt_model(input_batch)
    last_token_logits= logits[:, -1, :]
    loss= torch.nn.functional.cross_entropy(input= last_token_logits, 
                                            target= target_batch)
    return loss

#Function to find the loss of complete loader
def loader_classification_loss(gpt_model, loader, device, num_batches= None):
    if len(loader) == 0:
        return float('nan')
    elif num_batches == None:
        num_batches= len(loader)
    else:
        num_batches= min(num_batches, len(loader))
    total_loss= 0
    for i, (input_batch, target_batch) in enumerate(loader):
        if i < num_batches:
            batch_loss= batch_classification_loss(gpt_model= gpt_model,
                                                  input_batch= input_batch,
                                                  target_batch= target_batch,
                                                  device= device)
            total_loss += batch_loss
        else:
            break
    loss_per_batch= total_loss / num_batches
    return loss_per_batch

def loader_classification_loss_v2(gpt_model, loader, device):
    num_batches= 0
    total_loss= 0
    for input_batch, target_batch in loader:
        batch_loss= batch_classification_loss(gpt_model= gpt_model, 
                                              input_batch= input_batch,
                                              target_batch= target_batch, 
                                              device= device)
        to
        num_batches+=1

    return 
#Function to evaluate model--->Find train and validation loader loss
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = loader_classification_loss(train_loader, model, device, num_batches=eval_iter)
        val_loss = loader_classification_loss(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

#This function will return the gpt model having all layers freezed except the last one
def freeze_model_layers(gpt_model, num_classes, got_configurations):
    for params in gpt_model.parameters():
        params.required_grad()= False
    gpt_model.out_head= torch.nn.Linear(in_features= got_configurations['emb_dim'],
                                         out_features= num_classes)
    for params in gpt_model.out_head.parameters():
        params.requires_grad()= True
    for params in gpt_model.final_norm.parameters():
        params.requires_grad()= True
    return gpt_model