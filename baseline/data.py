import torch

from torchtext import data
from torchtext.data import Dataset
from torchtext import datasets
from torchtext.vocab import GloVe

from nltk import word_tokenize
import dill

import os

class SNLIDataset(Dataset):
    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.premise), len(ex.hypothesis))

# preprocess SNLI corpus to save time and give train, dev, test sets
class SNLI(object):
    def __init__(self, batch_size=4, gpu=torch.device('cuda')):
        # set file name for train dev test sets
        self.snli_split_path_lst = ['./data/snli_split/train', './data/snli_split/dev', './data/snli_split/test']

        # set data field for text and label
        self.TEXT = data.Field(batch_first=True, include_lengths=True, tokenize=word_tokenize, lower=True)
        self.LABEL = data.Field(sequential=False, unk_token=None)

        # split corpus
        if self.if_splited():
            # if already splited, load local sets
            fields = {'premise': self.TEXT, 'hypothesis': self.TEXT, 'label': self.LABEL}
            self.train, self.dev, self.test = self.load_split_datasets(fields)
        else:
            # split corpus to train, dev, test sets and save them to local
            self.train, self.dev, self.test = datasets.SNLI.splits(self.TEXT, self.LABEL, root='data')
            self.save_splited_sets(self.train, self.dev, self.test)

        # build vocab for corpus
        if os.path.exists('./data/snli_split/text_vocab') and os.path.exists('./data/snli_split/label_vocab'):
            # if local vocab exists, load local vocab into model
            with open('./data/snli_split/text_vocab', 'rb')as f:
                self.TEXT.vocab = dill.load(f)
            with open('./data/snli_split/label_vocab', 'rb')as f:
                self.LABEL.vocab = dill.load(f)
        else:
            # build vocab for corpus and save it to local
            self.TEXT.build_vocab(self.train, self.dev, self.test, vectors=GloVe(name='840B', dim=300))
            self.LABEL.build_vocab(self.train)
            with open('./data/snli_split/text_vocab', 'wb')as f:
                dill.dump(self.TEXT.vocab, f)
            with open('./data/snli_split/label_vocab', 'wb')as f:
                dill.dump(self.LABEL.vocab, f)

        # generate batch iterator
        self.train_iter, self.dev_iter, self.test_iter =  data.BucketIterator.splits((self.train, self.dev, self.test), batch_size=batch_size, device=gpu)

    # check local train, dev, test sets
    def if_splited(self):
        for path in self.snli_split_path_lst:
            if not os.path.exists(path):
                return False
        return True

    # load dataset from local
    def load_split_datasets(self, fields):
        # load from local
        with open('./data/snli_split/train', 'rb')as f:
            train_examples = dill.load(f)
        with open('./data/snli_split/dev', 'rb')as f:
            dev_examples = dill.load(f)
        with open('./data/snli_split/test', 'rb')as f:
            test_examples = dill.load(f)

        # recover
        train = SNLIDataset(examples=train_examples, fields=fields)
        dev = SNLIDataset(examples=dev_examples, fields=fields)
        test = SNLIDataset(examples=test_examples, fields=fields)
        return train, dev, test

    # save datasets to local
    def save_splited_sets(self, train, dev, test):
        # save to local
        with open('./data/snli_split/train', 'wb')as f:
            dill.dump(train.examples, f)
        with open('./data/snli_split/dev', 'wb')as f:
            dill.dump(dev.examples, f)
        with open('./data/snli_split/test', 'wb')as f:
            dill.dump(test.examples, f)