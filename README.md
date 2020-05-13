
# CSCI 544 Final Project - DR-BiLSTM Implementation
Attempted implementation of the paper "DR-BiLSTM: Dependent Reading Bidirectional LSTM for Natural Language Inference". 


## File Structures
best.pth - pretrained model used for testing

*.pkl - Preprocessed data files. Google Colab was used for preprocessing the dataset and training the model.

snli_training(_local).json - config files for model training. Since preprocessed files are included in the repository, we only include config files for training in case one wants to modify the parameters and re-train the model.

train_snli.py - script to train the model.

test_snli.py - script to test the model.

drlstm folder - model definition and util functions

## Getting Started
To test the model, run

    python test_snli.py
To train a new model, modify corresponding parameters in snli_training_local.json and run

    python train_snli.py
