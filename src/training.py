import os
import sys
import numpy as np
from time import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(1, str(Path(__file__).parent.parent / "src"))
from metrics import LossLog, AccLog


def train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval, mode, losses, accuracies):
    model.train()
    interval_time = time()
    for batch_idx, batch_data in enumerate(train_loader):
        optimizer.zero_grad()

        if mode == 'clf':
            batch_data, batch_label = batch_data
            batch_data = batch_data.to(device)
            outputs = model(batch_data)
            batch_label = batch_label.to(device)
            accuracies.update_acc(outputs, batch_label, 'train')
            loss_outputs = loss_fn(outputs, batch_label)
            bs = batch_data.shape[0]
        elif mode == 'siam':
            batch_data = tuple(d.to(device) for d in batch_data)
            outputs = model(*batch_data)
            loss_outputs = loss_fn(*outputs)
            bs = len(batch_data[0])
        else:
            raise ValueError

        losses.update_loss(loss_outputs.item(), 'train')
        loss_outputs.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}, time {} sec'.format(
                batch_idx * bs, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), losses.compute_average_loss('train'), time() - interval_time)
            print(message)
            interval_time = time()

    return losses, accuracies


def test_epoch(val_loader, model, loss_fn, device, mode, losses, accuracies):
    with torch.no_grad():
        model.eval()
        for batch_idx, batch_data in enumerate(val_loader):
            if mode == 'clf':
                batch_data, batch_label = batch_data
                batch_data = batch_data.to(device)
                outputs = model(batch_data)
                batch_label = batch_label.to(device)
                accuracies.update_acc(outputs, batch_label, 'test')
                loss_outputs = loss_fn(outputs, batch_label)
            elif mode == 'siam':
                batch_data = tuple(d.to(device) for d in batch_data)
                outputs = model(*batch_data)
                loss_outputs = loss_fn(*outputs)
            else:
                raise ValueError
            losses.update_loss(loss_outputs.item(), 'test')
    return losses, accuracies


def fit(mode: str, train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, device, log_interval,
        save_path: str, exp_name: str, batch_size, emb_size, start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """

    losses = LossLog()
    accuracies = AccLog(device=device)

    for epoch in range(0, start_epoch):
        scheduler.step()

    ep_log = []
    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        losses, accuracies = train_epoch(train_loader, model, loss_fn, optimizer,
                                         device, log_interval, mode, losses, accuracies)
        # Test stage
        losses, accuracies = test_epoch(val_loader, model, loss_fn, device, mode, losses, accuracies)
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
        pd.DataFrame(data=ep_log, columns=columns).to_csv(os.path.join(save_path, exp_name, 'ep_log.csv'),
                                                          index=False)
