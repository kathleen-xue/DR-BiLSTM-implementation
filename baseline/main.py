from data import SNLI
from model import Bowman
from utils import bowman_train

import torch
import torch.nn as nn

# set device for tensor as gpu
device = torch.device('cuda')

# do SNLI corpus preprocessing and initial Bowman model
snli = SNLI(batch_size=64, gpu=device)
model = Bowman(snli.TEXT.vocab)

if __name__ == "__main__":
    # move model to gpu
    model.to(device)
    # set loss function as cross entropy loss
    criterion = nn.CrossEntropyLoss()
    # set optimize function as AdamDelta SGD
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.1)

    # train the model
    bowman_train(model, snli, criterion, optimizer, epoch_num=100)