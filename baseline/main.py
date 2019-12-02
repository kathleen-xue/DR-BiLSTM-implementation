from data import SNLI
from model import Bowman
from utils import bowman_train

import torch
import torch.nn as nn

device = torch.device('cuda')
snli = SNLI(batch_size=8, gpu=device)
model = Bowman(snli.TEXT.vocab)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(model.parameters(), lr=0.1)

bowman_train(model, snli, criterion, optimizer, epoch_num=10)