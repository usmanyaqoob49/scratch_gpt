import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.classification_finetuning.utils import balance_dataset, class_mapping, train_val_test_split, map_classes, classify_text_example, convert_label_to_class
import pandas as pd
from src.classification_finetuning.data_loader import create_data_loaders
from src.data_preparation.utils import gpt_tokenizer
from src.classification_finetuning.utils import get_gpt_2_openai
from src.pretraining.generate_text import generate_diverse
from src.pretraining.utils import text_to_tokens, tokens_to_text
from src.pretraining.utils import gpt_2_124m_configurations
from src.gpt.utils import generate_text
from src.classification_finetuning.train import train
import torch

data_path= './data/processed/sms_spam_dataset/sms_spam_ham_data.csv'
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
train_dataset_loader, class_to_label_dict= create_data_loaders(
    dataset= train_set,
    text_col_name= 'sentence',
    class_col_name= 'emotion',
    batch_size= 8,
    num_workers= 0,
    shuffle= True,
    drop_last= True
)
validation_dataset_loader, class_to_label_dict= create_data_loaders(
    dataset=validation_set,
    text_col_name= 'sentence',
    class_col_name= 'emotion',
    batch_size= 8,
    num_workers= 0,
    shuffle= False,
    drop_last= False
    )
test_dataset_loader, class_to_label_dict= create_data_loaders(
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
print('Class to label mapping dictionary: ', class_to_label_dict)
print('-'*50)


#-------Testing function that returns gpt-2 model with laoded weights of openai
gpt_2_tokenizer= gpt_tokenizer()
gpt_2_model= get_gpt_2_openai()
sample_text= (
    "What is emotion in the sentence: "
    "I am feeling happy!"
)
output_token_ids= generate_text(gpt_model= gpt_2_model,
                            idx= text_to_tokens(tokenizer= gpt_2_tokenizer,
                                                text= sample_text),
                            max_new_tokens= 25,
                            context_size= gpt_2_124m_configurations['context_length'])
print("Output of loaded gpt-2 model before Classification finetuning: ", tokens_to_text(tokenizer= gpt_2_tokenizer,
                                                       tokens_ids= output_token_ids))
print('-'*50)

#------Testing training function that performs classification finetuning
batch_size= 5
num_epochs= 1
finetuned_model, training_accuray, validation_accuracy, training_loss, validation_loss, examples_seen= train(training_loader= train_dataset_loader,
                                                                                            validation_loader= validation_dataset_loader,
                                                                                            num_epochs= num_epochs,
                                                                                            eval_frequency= 50,
                                                                                            batch_size= batch_size)
torch.save(finetuned_model.state_dict(), f"./models/classification_finetuned/class_finetuned_{num_epochs}")
print("Training Accuracy: ", training_accuray)
print("Validation Accuracy: ", validation_accuracy)
print("Training Loss: ", training_loss)
print('Validation Loss: ', validation_loss)
print('-'*50)

#-------Testing prediction function and reverse labeling from label to class name function
text_sample= "Hello you have won the price!"
predicted_label= classify_text_example(finetuned_model= finetuned_model,
                                       text_example= text_sample,
                                       tokenizer= gpt_2_tokenizer)
print('Label predicted by the finetuned model: ', predicted_label)
print('Class Name predicted by the finetuned model: ', convert_label_to_class(class_mapping_dict= class_to_label_dict, label= predicted_label))