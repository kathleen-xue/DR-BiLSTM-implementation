import numpy as np
import torch.nn as nn
import torch
import os
import zipfile
import pickle
import fnmatch
import json
import string

from collections import Counter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import time
from tqdm import tqdm
from drlstm.utils import correct_predictions


from torch.utils.data import DataLoader
from drlstm.data import NLIDataset
from drlstm.model import DRLSTM

import matplotlib.pyplot as plt

"""
Utility functions for training and validating models.
"""
def train(model,
          dataloader,
          optimizer,
          criterion,
          epoch_number,
          max_gradient_norm):

    # Switch the model to train mode.
    model.train()
    device = model.device

    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0

    tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, batch in enumerate(tqdm_batch_iterator):
        batch_start = time.time()

        # Move input and output data to the GPU if it is used.
        premises = batch["premise"].to(device)
        premises_lengths = batch["premise_length"].to(device)
        hypotheses = batch["hypothesis"].to(device)
        hypotheses_lengths = batch["hypothesis_length"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits, probs = model(premises,
                              premises_lengths,
                              hypotheses,
                              hypotheses_lengths)
        loss = criterion(logits, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()

        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        correct_preds += correct_predictions(probs, labels)

        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}"\
                      .format(batch_time_avg/(batch_index+1),
                              running_loss/(batch_index+1))
        tqdm_batch_iterator.set_description(description)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader.dataset)

    return epoch_time, epoch_loss, epoch_accuracy


def validate(model, dataloader, criterion):
    # Switch to evaluate mode.
    model.eval()
    device = model.device

    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0

    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch in dataloader:
            # Move input and output data to the GPU if one is used.
            premises = batch["premise"].to(device)
            premises_lengths = batch["premise_length"].to(device)
            hypotheses = batch["hypothesis"].to(device)
            hypotheses_lengths = batch["hypothesis_length"].to(device)
            labels = batch["label"].to(device)

            logits, probs = model(premises,
                                  premises_lengths,
                                  hypotheses,
                                  hypotheses_lengths)
            loss = criterion(logits, labels)

            running_loss += loss.item()
            running_accuracy += correct_predictions(probs, labels)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))

    return epoch_time, epoch_loss, epoch_accuracy

"""
Train the model on the preprocessed SNLI dataset.
"""
# Aurelien Coet, 2018.



def main(train_file,
         valid_file,
         embeddings_file,
         target_dir,
         hidden_size=300,
         dropout=0.5,
         num_classes=3,
         epochs=1,
         batch_size=32,
         lr=0.0004,
         patience=5,
         max_grad_norm=10.0,
         checkpoint=None):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(20 * "=", " Preparing for training ", 20 * "=")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # -------------------- Data loading ------------------- #
    print("\t* Loading training data...")
    with open(train_file, "rb") as pkl:
        train_data = NLIDataset(pickle.load(pkl))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    print("\t* Loading validation data...")
    with open(valid_file, "rb") as pkl:
        valid_data = NLIDataset(pickle.load(pkl))

    valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)

    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    with open(embeddings_file, "rb") as pkl:
        embeddings = torch.tensor(pickle.load(pkl), dtype=torch.float)\
                     .to(device)

    model = DRLSTM(embeddings.shape[0],
                 embeddings.shape[1],
                 hidden_size,
                 embeddings=embeddings,
                 dropout=dropout,
                 num_classes=num_classes,
                 device=device).to(device)

    # -------------------- Preparation for training  ------------------- #
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="max",
                                                           factor=0.5,
                                                           patience=0)

    best_score = 0.0
    start_epoch = 1

    # Data for loss curves plot.
    epochs_count = []
    train_losses = []
    valid_losses = []

    # Continuing training from a checkpoint if one was given as argument.
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]

        print("\t* Training will continue on existing model from epoch {}..."
              .format(start_epoch))

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]

    # Compute loss and accuracy before starting (or resuming) training.
    _, valid_loss, valid_accuracy = validate(model,
                                             valid_loader,
                                             criterion)
    print("\t* Validation loss before training: {:.4f}, accuracy: {:.4f}%"
          .format(valid_loss, (valid_accuracy*100)))

    # -------------------- Training epochs ------------------- #
    print("\n",
          20 * "=",
          "Training model on device: {}".format(device),
          20 * "=")

    patience_counter = 0
    for epoch in range(start_epoch, epochs+1):
        epochs_count.append(epoch)

        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = train(model,
                                                       train_loader,
                                                       optimizer,
                                                       criterion,
                                                       epoch,
                                                       max_grad_norm)

        train_losses.append(epoch_loss)
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"
              .format(epoch_time, epoch_loss, (epoch_accuracy*100)))

        print("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = validate(model,
                                                          valid_loader,
                                                          criterion)

        valid_losses.append(epoch_loss)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n"
              .format(epoch_time, epoch_loss, (epoch_accuracy*100)))

        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_accuracy)

        # Early stopping on validation accuracy.
        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            # Save the best model. The optimizer is not saved to avoid having
            # a checkpoint file that is too heavy to be shared. To resume
            # training from the best model, use the 'drlstm_*.pth.tar'
            # checkpoints instead.
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "best_score": best_score,
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                       os.path.join(target_dir, "best.pth.tar"))

        # Save the model at each epoch.
        print ('Saving model to')
        print (os.path.join(target_dir, "drlstm_{}.pth.tar".format(epoch)))
        torch.save({"epoch": epoch,
                    "model": model.state_dict(),
                    "best_score": best_score,
                    "optimizer": optimizer.state_dict(),
                    "epochs_count": epochs_count,
                    "train_losses": train_losses,
                    "valid_losses": valid_losses},
                   os.path.join(target_dir, "drlstm_{}.pth.tar".format(epoch)))

        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break

    # Plotting of the loss curves for the train and validation sets.
    plt.figure()
    plt.plot(epochs_count, train_losses, "-r")
    plt.plot(epochs_count, valid_losses, "-b")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["Training loss", "Validation loss"])
    plt.title("Cross entropy loss")
    plt.show()

if __name__ == '__main__':
    default_config = "snli_training.json"

    script_dir = './'
    config = default_config
    checkpoint = None

    if config == default_config:
        config_path = os.path.join(script_dir, config)
    else:
        config_path = config

    with open(os.path.normpath(config_path), 'r') as config_file:
        config = json.load(config_file)

    main(os.path.normpath(os.path.join(script_dir, config["train_data"])),
          os.path.normpath(os.path.join(script_dir, config["valid_data"])),
          os.path.normpath(os.path.join(script_dir, config["embeddings"])),
          os.path.normpath(os.path.join(script_dir, config["target_dir"])),
          config["hidden_size"],
          config["dropout"],
          config["num_classes"],
          config["epochs"],
          config["batch_size"],
          config["lr"],
          config["patience"],
          config["max_gradient_norm"],
          checkpoint)
