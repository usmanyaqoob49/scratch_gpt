# scratch gpt│
├── data/                           # Directory for raw and processed datasets
│   ├── raw/                        # Raw datasets
│   ├── processed/                  # Preprocessed datasets
│   └── README.md                   # Notes on data sources and structure
│
├── src/                            # Source code
│   ├── data_preparation/           # Module for data preparation
│   │   ├── _init_.py
│   │   ├── preprocess.py           # Functions for data preprocessing
│   │   └── utils.py                # Utility functions for data handling
│   │
│   ├── attention/                  # Attention mechanism module
│   │   ├── _init_.py
│   │   ├── attention.py            # Scratch code for attention mechanisms
│   │   └── utils.py                # Utilities for attention calculations
│   │
│   ├── gpt_model/                  # GPT model module
│   │   ├── _init_.py
│   │   ├── model.py                # Model architecture and related code
│   │   └── config.py               # Model configuration and hyperparameters
│   │
│   ├── pretraining/                # Pretraining module
│   │   ├── _init_.py
│   │   ├── trainer.py              # Code for model pretraining
│   │   └── utils.py                # Utilities for training (e.g., logging)
│   │
│   ├── text_generation/            # Text generation module
│   │   ├── _init_.py
│   │   ├── generator.py            # Code for text generation
│   │   └── utils.py                # Utilities for text generation
│   │
│   └── _init_.py                 # Makes src a package
│
├── tests/                          # Unit and integration tests
│   ├── test_data_preparation.py    # Tests for data preparation module
│   ├── test_attent