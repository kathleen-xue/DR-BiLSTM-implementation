import torch
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
    # switch to train mode
    model.train()
    for epoch in range(epoch_num):
        running_loss = 0.0
        epoch_loss = 0.0
        i = 0
        for batch in snli.train_iter:
            i += 1
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

            # add loss for the batch
            running_loss += loss.item()
            epoch_loss += loss.item()
            if i % 1000 == 999:
                # print average running loss for each 1000 batch
                print('[%d, %5d] loss: %.3f' % (epoch, i + 1, running_loss / 1000))
                running_loss = 0.0
        # print average loss for the epoch
        print('epoch %d loss: %.3f\n' % (epoch, epoch_loss / (i + 1)))
        # save average loss for the epoch
        record.write(b'%f\n' % (epoch_loss / (i + 1)))
        # save trained model after the epoch
        torch.save(model, './model/bowman_%d.pth'% (epoch))

    # save final trained model
    torch.save(model, './model/bowman_final.pth')


def bowman_eval(model, dataset):
    # model - model
    # dataset - evaluation set
    snli = dataset

    # switch to evaluation mode
    model.eval()

    c_count = 0.
    t_count = 0.
    for batch in snli.train_iter:
        # get data
        premise, _ = batch.premise
        hypothesis, _ = batch.hypothesis
        label = batch.label

        # do predict
        output = model(premise, hypothesis)
        predict = torch.argmax(output, dim=1)

        batch_size = predict.shape
        # total number
        t_count += batch_size[0]
        # correct number
        c_count += int(torch.sum(predict == label))
    # calcualte the accuracy and print it out
    print("Train acc.: %f" % (c_count / t_count))
    
    c_count = 0.
    t_count = 0.
    for batch in snli.test_iter:
        premise, _ = batch.premise
        hypothesis, _ = batch.hypothesis
        label = batch.label
        output = model(premise, hypothesis)
        predict = torch.argmax(output, dim=1)
        batch_size = predict.shape
        t_count += batch_size[0]
        c_count += int(torch.sum(predict == label))
    print("Test acc.: %f" % (c_count / t_count))


if __name__ == "__main__":
    model = torch.load("./model/bowman_final.pt")
    device = torch.device('cuda')
    snli = SNLI(batch_size=16, gpu=device)
    bowman_eval(model, snli)