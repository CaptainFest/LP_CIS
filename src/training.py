import os
import sys
import numpy as np
from time import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(1, str(Path(__file__).parent.parent / "src"))
from metrics import LossLog, AccLog


def fit_siam(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, device, log_interval,
             save_path: str, exp_name:str, batch_size, emb_size, start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """

    losses = LossLog()

    for epoch in range(0, start_epoch):
        scheduler.step()

    ep_log = []
    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        losses = train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval, losses)
        train_loss = losses.compute_average_loss('train')
        # Val stage
        losses = test_epoch(val_loader, model, loss_fn, device, losses)
        val_loss = losses.compute_average_loss('test')
        # Test stage
        print(f"Epoch: {epoch + 1}/{n_epochs}. Train set: Average loss: {train_loss:.4f}")
        print(f"Epoch: {epoch+1}/{n_epochs}. Validation set: Average loss: {val_loss:.4f}")

        ep_log.append([epoch, train_loss, val_loss])
        losses.reset()

    if save_path is not None:
        os.makedirs(os.path.join(save_path, exp_name), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_path,
                                                    exp_name,
                                                    f"triplenet_ep{n_epochs}_bs{batch_size}_emb{emb_size}.pth"))
        pd.DataFrame(data=ep_log, columns=['ep', 'train_loss', 'val_loss']).to_csv(os.path.join(save_path,'ep_log.csv'))


def train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval, losses):
    model.train()
    interval_time = time()
    for batch_idx, batch_data in enumerate(train_loader):
        if device == 'cuda':
            batch_data = tuple(d.cuda() for d in batch_data)

        optimizer.zero_grad()
        outputs = model(*batch_data)

        loss_outputs = loss_fn(*outputs)
        losses.update_loss(loss_outputs.item(), 'train')
        loss_outputs.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}, time {} sec'.format(
                batch_idx * len(batch_data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), losses.compute_average_loss('train'), time() - interval_time)
            print(message)
            interval_time = time()

    return losses


def test_epoch(val_loader, model, loss_fn, device, losses):
    with torch.no_grad():
        losses.reset()
        model.eval()
        for batch_idx, batch_data in enumerate(val_loader):
            if device == 'cuda':
                batch_data = tuple(d.cuda() for d in batch_data)
            outputs = model(*batch_data)
            loss_outputs = loss_fn(*outputs)
            losses.update_loss(loss_outputs.item(), 'test')
    return losses

def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, device, log_interval,
        save_path: str, exp_name:str, batch_size, emb_size, mode:str='siam', start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """

    losses = LossLog()
    accuracies = AccLog()

    for epoch in range(0, start_epoch):
        scheduler.step()

    ep_log = []
    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        losses, accuracies = train_epoch(train_loader, model, loss_fn, optimizer,
                                         device, log_interval, losses, accuracies)
        # Test stage
        losses, accuracies = test_epoch(val_loader, model, loss_fn, device, losses, accuracies)
        train_loss = losses.compute_average_loss('train')
        test_loss = losses.compute_average_loss('test')
        temp_log = [epoch, train_loss, test_loss]

        train_mess = f"Epoch: {epoch + 1}/{n_epochs}. Train set: Average loss: {train_loss:.4f}"
        test_mess = f"Epoch: {epoch + 1}/{n_epochs}. Test set: Average loss: {test_loss:.4f}"

        if mode == 'clf':
            train_acc = accuracies.compute_average_acc('train')
            test_acc = accuracies.compute_average_acc('test')
            train_mess += f"/tAcc {train_acc:.3f}"
            test_mess += f"/tAcc {test_acc:.3f}"
            temp_log.extend([train_acc, test_acc])
            accuracies.reset()

        print(train_mess)
        print(test_mess)

        ep_log.append(temp_log)
        losses.reset()

    if save_path is not None:
        os.makedirs(os.path.join(save_path, exp_name), exist_ok=True)
        if mode == 'siam':
            columns = ['ep', 'train_loss', 'val_loss']
            model_name = f"triplenet_ep{n_epochs}_bs{batch_size}_emb{emb_size}.pth"
        elif mode == 'clf':
            columns = ['ep', 'train_loss', 'val_loss', 'train_acc', 'val_acc']
            model_name = f"clfnet_ep{n_epochs}_bs{batch_size}_emb{emb_size}.pth"
        else:
            raise ValueError
        torch.save(model.state_dict(), os.path.join(save_path, exp_name, model_name))
        pd.DataFrame(data=ep_log, columns=columns).to_csv(os.path.join(save_path,'ep_log.csv'))
