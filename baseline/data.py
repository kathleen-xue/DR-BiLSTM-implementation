import torch

from torchtext import data
from torchtext.data import Dataset
from torchtext import datasets
from torchtext.vocab import GloVe

from nltk import word_tokenize
import dill
from config import *

class SNLIDataset(Dataset):
    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(
            len(ex.premise), len(ex.hypothesis))


class SNLI(object):
    def __init__(self, batch_size=4, gpu=torch.device(torch.cuda.current_device())):
        self.TEXT = data.Field(batch_first=True,
                               include_lengths=True,
                               tokenize=word_tokenize,
                               lower=True)

        self.LABEL = data.Field(sequential=False, unk_token=None)

        # Split Dataset
        if self.if_split_already():
            print('Loading splited data set...')
            fields = {'premise': self.TEXT, 'hypothesis': self.TEXT, 'label': self.LABEL}
            self.train, self.dev, self.test = self.load_split_datasets(fields)
        else:
            print('No local data set detected, spliting...')
            self.train, self.dev, self.test = datasets.SNLI.splits(self.TEXT, self.LABEL, root='data')
            self.dump_examples(self.train, self.dev, self.test)


        # Create Vocab
        print('Building Vocab...')
        # self.TEXT.build_vocab(self.train, self.dev, self.test, vectors=GloVe(name='840B', dim=300))
        # self.LABEL.build_vocab(self.train)
        if os.path.exists(snli_text_vocab_path) and os.path.exists(snli_label_vocab_path):
            print('Loading local Vocab...')
            with open(snli_text_vocab_path, 'rb')as f:
                self.TEXT.vocab = dill.load(f)
            with open(snli_label_vocab_path, 'rb')as f:
                self.LABEL.vocab = dill.load(f)
        else:
            print('No local Vocab detected, building...')
            self.TEXT.build_vocab(self.train, self.dev, self.test, vectors=GloVe(name='840B', dim=300))
            self.LABEL.build_vocab(self.train)
            with open(snli_text_vocab_path, 'wb')as f:
                dill.dump(self.TEXT.vocab, f)
            with open(snli_label_vocab_path, 'wb')as f:
                dill.dump(self.LABEL.vocab, f)


        # Generate batch iterator
        print('Generating batch iter...')
        self.train_iter, self.dev_iter, self.test_iter = \
            data.BucketIterator.splits((self.train, self.dev, self.test),
                                       batch_size=batch_size,
                                       device=gpu)

    def if_split_already(self):
        for path in snli_split_path_lst:
            if not os.path.exists(path):
                return False
        return True

    # Load dataset from local
    def load_split_datasets(self, fields):
        # Loading examples
        with open(snli_train_examples_path, 'rb')as f:
            train_examples = dill.load(f)
        with open(snli_dev_examples_path, 'rb')as f:
            dev_examples = dill.load(f)
        with open(snli_test_examples_path, 'rb')as f:
            test_examples = dill.load(f)

        # Recover dataset
        train = SNLIDataset(examples=train_examples, fields=fields)
        dev = SNLIDataset(examples=dev_examples, fields=fields)
        test = SNLIDataset(examples=test_examples, fields=fields)
        return train, dev, test

    # Save to local
    def dump_examples(self, train, dev, test):
        # Save examples
        if not os.path.exists(snli_train_examples_path):
            with open(snli_train_examples_path, 'wb')as f:
                dill.dump(train.examples, f)
        if not os.path.exists(snli_dev_examples_path):
            with open(snli_dev_examples_path, 'wb')as f:
                dill.dump(dev.examples, f)
        if not os.path.exists(snli_test_examples_path):
            with open(snli_test_examples_path, 'wb')as f:
                dill.dump(test.examples, f)
