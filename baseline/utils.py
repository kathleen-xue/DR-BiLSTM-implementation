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
        runlos = 0.0
        i = 0
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
            runlos += loss.item()
            i+=1
            if i % 1000 == 999:
                print(runlos/1000)
                runlos = 0.0
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

def analysis_eval(model, dataset, set_name, neg_set, qua_set, bel_set):

    # model - model
    # dataset - evaluation set
    snli = dataset

    # switch to evaluation mode
    model.eval()

    batch_iter = None

    if set_name == "Dev":
        batch_iter = snli.dev_iter
    elif set_name == "Test":
        batch_iter = snli.test_iter
    else:
        return
    
    correct_dict = [0.0] * 9 
    total_dict = [0.0] * 9
    for batch in batch_iter:
        # get data
        premise, _ = batch.premise
        hypothesis, _ = batch.hypothesis
        label = batch.label

        # do predict
        output = model(premise, hypothesis)
        predict = torch.argmax(output, dim=1)

        pre_tokens = set(premise[0].cpu().numpy().tolist())
        hyp_tokens = set(hypothesis[0].cpu().numpy().tolist())

        overlap_num = float(len(set.intersection(pre_tokens, hyp_tokens)))
        token_total = float(len(set.union(pre_tokens, hyp_tokens)))

        # check for overlap
        if (overlap_num/token_total > 0.7):
            total_dict[0] += 1
            if label[0] == predict[0]:
                correct_dict[0] += 1
        elif (overlap_num/token_total < 0.3):
            total_dict[2] += 1
            if label[0] == predict[0]:
                correct_dict[2] += 1
        else:
            total_dict[1] += 1
            if label[0] == predict[0]:
                correct_dict[1] += 1
                
        # check for sentence length
        if (premise[0].shape[0] > 20 or hypothesis[0].shape[0] > 20):
            total_dict[3] += 1
            if label[0] == predict[0]:
                correct_dict[3] += 1
        elif (premise[0].shape[0] < 5 or hypothesis[0].shape[0] < 5):
            total_dict[5] += 1
            if label[0] == predict[0]:
                correct_dict[5] += 1
        else:
            total_dict[4] += 1
            if label[0] == predict[0]:
                correct_dict[4] += 1

        # check for cotaning specific words
        if (len(set.intersection(pre_tokens, neg_set))> 0 or len(set.intersection(hyp_tokens, neg_set))> 0):
            total_dict[6] += 1
            if label[0] == predict[0]:
                correct_dict[6] += 1
        if (len(set.intersection(pre_tokens, qua_set))> 0 or len(set.intersection(hyp_tokens, qua_set))> 0):
            total_dict[7] += 1
            if label[0] == predict[0]:
                correct_dict[7] += 1
        if (len(set.intersection(pre_tokens, bel_set))> 0 or len(set.intersection(hyp_tokens, bel_set))> 0):
            total_dict[8] += 1
            if label[0] == predict[0]:
                correct_dict[8] += 1

    print("Overlap:")
    # High: > 70%, Regular: 30%-70%, Low: < 30%
    print("High: %.3f\t\t Regular: %.3f\t\t Low: %.3f" % (correct_dict[0] / total_dict[0], correct_dict[1] / total_dict[1], correct_dict[2] / total_dict[2]))
    
    print("Sentence Length:")
    # Long: > 20, Regular: 5-20, Short: < 5
    print("Long: %.3f\t\t Regular: %.3f\t\t Short: %.3f" % (correct_dict[3] / total_dict[3], correct_dict[4] / total_dict[4], correct_dict[5] / total_dict[5]))
    
    print("Contain Specific Words:")
    print("Negation: %.3f\t\t Quantifier: %.3f\t Belief: %.3f" % (correct_dict[6] / total_dict[6], correct_dict[7] / total_dict[7], correct_dict[8] / total_dict[8]))


if __name__ == "__main__":
    device = torch.device('cuda')
    snli = SNLI(batch_size=1, gpu=device)
    model = Bowman(snli.TEXT.vocab)
    model.load_state_dict(torch.load("./model/bowman_48.pth"))
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # build list for type of specific words
    neg = ["no", "not", "none", "nobody", "nothing", "neither", "nowhere", "never", "hardly", "barely", "doesnt", "isnt", "wasnt", "shouldnt", "wouldnt", "couldnt", "wont", "cant", "dont"]
    qua = ["much", "enough", "more", "most", "less", "least", "no", "none", "some", "any", "many", "few", "several", "almost", "nearly"]
    bel = ["know", "believe", "understand", "doubt", "think", "suppose", "recognize", "forget", "remember", "imagine", "mean", "agree", "disagree", "deny", "promise"]

    neg_set, qua_set, bel_set = [], [], []

    for token in neg:
        if token in snli.TEXT.vocab.stoi:
            neg_set.append(snli.TEXT.vocab.stoi[token])
    for token in qua:
        if token in snli.TEXT.vocab.stoi:
            qua_set.append(snli.TEXT.vocab.stoi[token])
    for token in bel:
        if token in snli.TEXT.vocab.stoi:
            bel_set.append(snli.TEXT.vocab.stoi[token])

    neg_set, qua_set, bel_set = set(neg_set), set(qua_set), set(bel_set)
    analysis_eval(model, snli, "Test", neg_set, qua_set, bel_set)
    acc, loss = bowman_eval(model, snli, "Train", criterion)
    acc, loss = bowman_eval(model, snli, "Test", criterion)