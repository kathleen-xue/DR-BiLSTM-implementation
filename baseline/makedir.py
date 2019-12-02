import os
from os.path import join

# claim directories for saving data and model
new_dir = ['./data', './model', './data/snli_split']

# if directories not exist, make new directory
for dir in new_dir:
    if not os.path.exists(dir):
        print('mkdir:', dir)
        os.mkdir(dir)