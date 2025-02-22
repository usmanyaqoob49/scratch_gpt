# scratch gpt
I tried my best to do functional coding so that anyone can use this project for their usecase by using their own datasets. I tired my best to keep it clean but I would love to see it more improved :) 

Everything is written using python and pytorch, no libraries or stuff. I tried to code everything from scratch. 

Project Directory details:

            - Data preparation codes that we need for using the coding the LLM like GPT-2 from scratch in src.data_preparation.
            - Scratch Codes of Scaled Self Attention, Causal Self Attention, Multihead Attention in src.attention_mechanism.
            - Transformer scratch code in src.transformer.
            - GPT-2 124M architecture developed from scratch using pytorch in src.gpt.
            - Functions and data loaders for pretraining of gpt-2 using any textual dataset in src.pretraining.
            - Code for classification finetuning of gpt-2 on any class dataset in src.classification_finetuning.
            - Code for instruction finetuning of gpt-2 on any instructions data set in src.instruction-finetuning.

All test files are included in tests folder, that clearly guide the overall use of the project.

GO through the project in this order to get most out of it:
    