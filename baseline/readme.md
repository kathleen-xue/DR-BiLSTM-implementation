#!/bin/bash

# Train Bowman's model on local device

# Build local directories for saving datasets
python3 makedir.py

# Train the model and save model in local drive
python3 main.py
# Plot the curve for loss and acc. in trainning if needed
python3 plotresult.py


# Get sample result for first 5 examples in test set
python3 test.py
# Get acc. for different annotations
python3 utils.py