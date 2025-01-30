import pandas as pd

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
    validation_set= data[train_split_index:validation_split_index]
    test_set= data[validation_split_index:]
    return train_set, validation_set, test_set
