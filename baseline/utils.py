import torch
import torch.nn as nn
from model import Bowman
from data import SNLI

def bowman_train(model, dataset, criterion, optimizer, epoch_num=5):
    # model - model
    # dataset - traning set
    # criterion - loss function
    # optimizer - optimize function
    # epoch_num - epoch number
    snli = dataset
    # file to record average loss for each epoch
    record = open("result.txt", "wb", buffering=0)
    for epoch in range(epoch_num):
        # switch to train mode
        model.train()

        for batch in snli.train_iter:
            # get data
            premise, _ = batch.premise
            hypothesis, _ = batch.hypothesis
            label = batch.label

            # zeros the parameters gradients
            optimizer.zero_grad()

            # forward + backward + optimize step
            output = model(premise, hypothesis)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

        train_acc, train_loss = bowman_eval(model, dataset, "Train", criterion)
        dev_acc, dev_loss = bowman_eval(model, dataset, "Dev", criterion)
        # print average loss for the epoch
        print('epoch %d train_loss: %.3f dev_loss: %.3f train_acc: %.3f dev_acc: %.3f' % (epoch, train_loss, dev_loss, train_acc, dev_acc))
        # save average loss for the epoch
        record.write(b'%f\t%f\t%f\t%f\n' % (train_loss, dev_loss, train_acc, dev_acc))
        # save trained model after the epoch
        torch.save(model.state_dict(), './model/bowman_%d.pth'% (epoch))

    # save final trained model
    torch.save(model.state_dict(), './model/bowman_final.pth')


def bowman_eval(model, dataset, set_name, criterion):
    # model - model
    # dataset - evaluation set
    snli = dataset

    # switch to evaluation mode
    model.eval()

    batch_iter = None

    if set_name == "Train":
        batch_iter = snli.train_iter
    elif set_name == "Dev":
        batch_iter = snli.dev_iter
    elif set_name == "Test":
        batch_iter = snli.test_iter
    else:
        return

    c_count = 0.
    t_count = 0.
    epoch_loss = 0.0
    for batch in batch_iter:
        # get data
        premise, _ = batch.premise
        hypothesis, _ = batch.hypothesis
        label = batch.label

        # do predict
        output = model(premise, hypothesis)
        predict = torch.argmax(output, dim=1)
        loss = criterion(output, label)
        batch_size = predict.shape

        epoch_loss += loss.item() * batch_size[0]

        # total number
        t_count += batch_size[0]
        # correct number
        c_count += int(torch.sum(predict == label))
    # calcualte the accuracy and print it out
    # print("%s acc.: %f" % (set_name, c_count / t_count))
    return c_count / t_count, epoch_loss / t_count

if __name__ == "__main__":
    model = Bowman()
    model.load_state_dict(torch.load("./model/bowman_final.pth"))
    device = torch.device('cuda')
    snli = SNLI(batch_size=16, gpu=device)
    criterion = nn.CrossEntropyLoss()
    acc, loss = bowman_eval(model, snli, "Train", criterion)
    print("Train acc.: %.3f, loss : %.3f" % (acc, loss))
    acc, loss = bowman_eval(model, snli, "Test", criterion)
    print("Test acc.: %.3f, loss : %.3f" % (acc, loss))