import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.classification_finetuning.utils import balance_dataset, class_mapping, train_val_test_split, map_classes
import pandas as pd
from src.classification_finetuning.data_loader import create_data_loaders
from src.data_preparation.utils import gpt_tokenizer
from src.classification_finetuning.gpt_model import get_gpt_2_openai
from src.pretraining.generate_text import generate_diverse
from src.pretraining.utils import text_to_tokens, tokens_to_text
from src.pretraining.utils import gpt_2_124m_configurations

data_path= './data/processed/emotion_dataset/combined_emotions_data.csv'
balance_data= balance_dataset(dataset_path= data_path, classes_column_name= 'emotion')
print(balance_data['emotion'].value_counts())
print('-'*50)

#------Testing random split function
data_df= pd.read_csv(data_path)
train_set, validation_set, test_set= train_val_test_split(data=data_df)
print('shape of train set: ', train_set.shape)
print('shape of validation set: ', validation_set.shape)
print('shape of test set: ', test_set.shape)
print('-'*50)

#-------Testing data loader class
train_dataset_loader= create_data_loaders(
    dataset= train_set,
    text_col_name= 'sentence',
    class_col_name= 'emotion',
    batch_size= 8,
    num_workers= 0,
    shuffle= True,
    drop_last= True
)
validation_dataset_loader= create_data_loaders(
    dataset=validation_set,
    text_col_name= 'sentence',
    class_col_name= 'emotion',
    batch_size= 8,
    num_workers= 0,
    shuffle= False,
    drop_last= False
    )
test_dataset_loader= create_data_loaders(
    dataset= test_set,
    text_col_name= 'sentence',
    class_col_name= 'emotion',
    batch_size= 8,
    num_workers= 0,
    shuffle= False,
    drop_last= False)

print("Train loader:")
for input_batch, target_batch in train_dataset_loader:
    pass

print("Input batch dimensions:", input_batch.shape)
print("Label batch dimensions", target_batch.shape)
print('-'*50)


#-------Testing function that returns gpt-2 model with laoded weights of openai
gpt_2_tokenizer= gpt_tokenizer()
gpt_2_model= get_gpt_2_openai()
sample_text= (
    "What is emotion in the sentence: "
    "I am feeling happy!"
)
print("Start Context sample for LLM: ", sample_text)
output_token_ids= generate_diverse(model= gpt_2_model,
                            idx= text_to_tokens(tokenizer= gpt_2_tokenizer,
                                                text= sample_text),
                            max_new_tokens= 25,
                            context_size= gpt_2_124m_configurations)
print("Output of loaded gpt-2 model: ", tokens_to_text(tokenizer= gpt_2_tokenizer,
                                                       tokens_ids= output_token_ids))
