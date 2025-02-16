from fastapi import FastAPI, Form
from pydantic import BaseModel
from src.pretraining.pretrain_gpt import pretrain_gpt


app= FastAPI()
#--------------API that will take data file path and will pretrain gpt-2 
@app.post("/pretrain")
async def train(
    file_path: str = Form(...),
    epochs: int = Form(...)):    
    train_losses, validation_losses, tokens_seen = pretrain_gpt(file_path, epochs)
    return train_losses, validation_losses


#------------API to load openAI weights in our gpt architecture using src.pretrain.loading_weights_into_gpt.py 













#-------------API to use the gpt model that has been loaded with the weights of gpt-2 above (text input text output)











#---------API that will take classification dataset, will process it (make loaders), and wll call train.py from src.classification finetuning 








#---------API that will take any text for the classification and it will use classify_text_example function for src.classificaiton_finetuning.utils to perform the classification on the text example
