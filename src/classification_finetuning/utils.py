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
    