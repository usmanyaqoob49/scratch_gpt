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

#Function to find the classification loss of the loader
def loader_classification_loss(data_loader, model, device, num_batches= None):
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
            correct_predictions+= 