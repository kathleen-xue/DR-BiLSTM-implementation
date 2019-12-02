import os
from os.path import join

new_dir = []
config_path = os.path.abspath(__file__)
project_dir_path = os.path.dirname(config_path)
data_dir_path = join(project_dir_path, 'data')
model_dir_path = join(project_dir_path, 'model')
snli_split_dir_path = join(data_dir_path, 'snli_split')
snli_train_examples_path = join(snli_split_dir_path, 'train_examples')
snli_dev_examples_path = join(snli_split_dir_path, 'dev_examples')
snli_test_examples_path = join(snli_split_dir_path, 'test_examples')

snli_split_path_lst = [snli_train_examples_path, snli_dev_examples_path, snli_test_examples_path]

snli_text_vocab_path = join(snli_split_dir_path, 'text_vocab')
snli_label_vocab_path = join(snli_split_dir_path, 'label_vocab')

new_dir.append(data_dir_path)
new_dir.append(snli_split_dir_path)
new_dir.append(model_dir_path)

for dir in new_dir:
    if not os.path.exists(dir):
        print('mkdir:', dir)
        os.mkdir(dir)