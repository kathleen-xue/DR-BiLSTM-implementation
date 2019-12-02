from data import SNLI
from model import Bowman
from utils import bowman_train

import torch
import torch.nn as nn

device = torch.device('cuda')
snli = SNLI(batch_size=16, gpu=device)
model = Bowman(snli.TEXT.vocab)


if __name__ == "__main__":
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.01)

    bowman_trai