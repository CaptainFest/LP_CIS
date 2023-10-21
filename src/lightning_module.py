import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models
from torch.optim import lr_scheduler, Adam

from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall, MulticlassPrecision, MulticlassF1Score, \
    MulticlassConfusionMatrix


class EmbedNetwork(nn.Module):
    def __init__(self, last_feat_num:int=2):
        super(EmbedNetwork, self).__init__()
        # get resnet model
        self.resnet = models.resnet18(weights=None)

        self.fc_in_features = self.resnet.fc.in_features

        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))

        # add linear layers
        self.last_feat_num = last_feat_num
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.last_feat_num),
        )

        # initialize the weights
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output


class LitTriplet(pl.LightningModule):
    def __init__(self, last_feat_num: int, loss_fn, online):
        super().__init__()
        # get resnet model
        self.embedding_net = EmbedNetwork(last_feat_num)
        self.online = online
        self.loss_fn = loss_fn

        # temporary variables
        self.output_losses = {'trian':[], 'valid': []}

    def get_embedding(self, x):
        output = self.embedding_net(x)
        return output

    def forward(self, input1, input2, input3):
        output1 = self.get_embedding(input1)
        output2 = self.get_embedding(input2)
        output3 = self.get_embedding(input3)

        return output1, output2, output3

    def training_step(self, batch_data, batch_idx):
        loss = self._shared_eval(batch_data, mode='train')
        return loss

    def validation_step(self, batch_data, batch_idx):
        loss = self._shared_eval(batch_data, mode='valid')

    def _shared_eval(self, batch, mode="train"):
        if self.online is not None:
            batch_data, batch_labels = batch
            outputs = self.get_embedding(batch_data)
            outputs = tuple([outputs, batch_labels])
        else:
            # sprint(type(batch), len(batch), batch[0].shape)
            outputs = self.forward(batch[0], batch[1], batch[2])
        loss = self.loss_fn(*outputs)
        # self.output_losses[mode].append(loss)
        self.log(f"{mode}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        self.save_model()

    def on_validation_epoch_end(self):
        pass
        # avg_loss =
        # self.logger.experiment.add_scalar("Loss/Train",
        #                                  avg_loss,
        #                                 self.current_epoch)
        # print(self.log)
        # self.validation_batch_preds.clear()

    def save_model(self):
        save_path = os.path.join(self.logger.save_dir, 'weights')
        os.makedirs(save_path, exist_ok=True)
        model_name = f'model_ep{self.current_epoch}.pt'
        model_fp = os.path.join(save_path, model_name)
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.embedding_net.state_dict(),
        }, model_fp)
        self.logger.experiment.log_model(model_name, model_fp)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-2)
        scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch"
            }
        }


def metrics_init(classes):
    accuracy = MulticlassAccuracy(classes, average='macro') #.to('cuda')
    precision = MulticlassPrecision(classes, average='macro') #.to('cuda')
    recall = MulticlassRecall(classes, average='macro') #.to('cuda')
    f1_score = MulticlassF1Score(classes, average='macro') #.to('cuda')
    cf_matrix = MulticlassConfusionMatrix(classes) #.to('cuda')
    metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1_score,
               'cf_matrix': cf_matrix}
    return metrics


def get_reg_names():
    data = pd.read_csv('../data/train_all_OCR_df.csv')
    reg_names = [data[data['reg_label'] == reg_label].iloc[0]['reg_name'] for reg_label in
                 np.unique(data['reg_label'])]
    return reg_names


class BaseClf(pl.LightningModule):
    def __init__(self, model, classes: int, save_folder: str, exp_name: str):
        super().__init__()
        self.embedding_net = model
        self.classes = classes
        self.loss_fn = nn.CrossEntropyLoss()

        self.save_folder = save_folder
        self.exp_name = exp_name

        self.batch_preds = {'train': [], 'valid': []}
        self.validation_batch_preds = []

        self.reg_names = get_reg_names()
        self.metrics = {'train': metrics_init(self.classes),
                        'valid': metrics_init(self.classes)}

    def forward(self, x):
        output = self.embedding_net(x)
        # scores = nn.functional.log_softmax(self.fc1(output), dim=-1)
        return output

    def _shared_eval(self, batch, mode="train"):
        batch_data, batch_labels = batch
        outputs = self.embedding_net(batch_data)
        # print(outputs.shape, batch_labels.shape)
        loss = self.loss_fn(outputs, batch_labels)
        log = {f"{mode}_loss": loss}
        for metric in self.metrics[mode]:
            self.metrics[mode][metric].update(outputs, batch_labels)
            if metric != 'cf_matrix':
                log[metric] = self.metrics[mode][metric].compute()
            if metric == 'accuracy':
                log["progress_bar"] = {metric: self.metrics[mode][metric].compute()}
        self.log_dict(log)
        return loss

    def training_step(self, batch_data, batch_idx):
        loss = self._shared_eval(batch_data, mode='train')
        return loss

    def validation_step(self, batch_data, batch_idx):
        loss = self._shared_eval(batch_data, mode='valid')
        return loss

    def on_train_epoch_end(self):
        results = {
            metric: self.metrics['train'][metric].compute() for metric in self.metrics['train']
            if metric != 'cf_matrix'
        }
        self.save_model()
        self.save_epoch_results(results, 'train')
        self.save_cf_matrix(self.metrics['train']['cf_matrix'])

    def on_validation_epoch_end(self):
        results = {metric: self.metrics['valid'][metric].compute() for metric in self.metrics['valid']}
        self.save_epoch_results(results, 'valid')
        self.save_cf_matrix(self.metrics['valid']['cf_matrix'])

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-2)
        scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch"
            }
        }

    def save_model(self):
        if self.save_path is not None:
            os.makedirs(os.path.join(self.save_folder, self.exp_name), exist_ok=True)
            model_name = f"basenet_ep{self.current_epoch}.pth"
            torch.save(self.embedding_net.state_dict(), os.path.join(self.save_path, self.exp_name, model_name))

    def save_epoch_results(self, results, mode: str):
        results['epoch'] = self.current_epoch
        res_path = os.path.join(self.save_folder, self.exp_name, f'ep_log_{mode}.csv')
        if os.path.exists(res_path):
            data = pd.read_csv(res_path)
            data = pd.concat([data, results], ignore_index=True)
        else:
            print(results)
            data = pd.DataFrame(data=results)
        data.to_csv(os.path.join(self.save_folder, self.exp_name, f'ep_log_{mode}.csv'), index=False)

    def save_cf_matrix(self, matrix):
        fig, ax = matrix.plot(labels=self.reg_names)
        fig.suptitle(f'{self.exp_name}_curep_{self.current_epoch}')
        fig.savefig(os.path.join(self.save_folder, self.exp_name, f"cf_matrix_ep{self.current_epoch}.png"))

class LitClf(pl.LightningModule):
    def __init__(self, embedding_net, emb_size: int, n_classes: int):
        super().__init__(embedding_net, n_classes)
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(emb_size, n_classes)

        self.batch_preds = {'train': [], 'valid': []}
        self.validation_batch_preds = []

        self.metrics = {'train': self.metrics_init(self.classes),
                        'valid': self.metrics_init(self.classes)}

    def forward(self, x):
        output = self.embedding_net.get_embedding(x)
        output = self.nonlinear(output)
        scores = nn.functional.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))

    def _shared_eval(self, batch, mode="train"):
        batch_data, batch_labels = batch
        outputs = self.embedding_net(batch_data)
        loss = nn.CrossEntropyLoss(outputs, batch_labels)
        log = {f"{mode}_loss": loss}
        for metric in self.metrics[mode]:
            self.metrics[mode][metric].update(outputs, batch_labels)
            log[metric] = self.metrics[mode][metric].compute()
            if metric == 'accuracy':
                log["progress_bar"] = {metric: self.metrics[mode][metric].compute()}
        self.log_dict(log)
        return loss

    def training_step(self, batch_data, batch_idx):
        loss = self._shared_eval(batch_data, mode='train')
        return loss

    def validation_step(self, batch_data, batch_idx):
        loss = self._shared_eval(batch_data, mode='valid')
        return loss

    def on_train_epoch_end(self):
        results = {metric: self.metrics['train'][metric].compute() for metric in self.metrics['train']}
        self.save_model()
        self.save_epoch_results(results, 'train')

    def on_validation_epoch_end(self):
        results = {metric: self.metrics['valid'][metric].compute() for metric in self.metrics['valid']}
        self.save_epoch_results(results, 'valid')

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-2)
        scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch"
            }
        }

    def save_model(self):
        if self.save_path is not None:
            os.makedirs(os.path.join(self.save_path, self.exp_name), exist_ok=True)
            model_name = f"basenet_ep{self.current_epoch}.pth"
            torch.save(self.embedding_net.state_dict(), os.path.join(self.save_path, self.exp_name, model_name))

    def save_epoch_results(self, results, mode: str):
        results['epoch'] = self.current_epoch
        res_path = os.path.join(self.save_path, self.exp_name, f'ep_log_{mode}.csv')
        if os.path.exists(res_path):
            data = pd.read_csv(res_path)
            data = pd.concat([data, results], ignore_index=True)
        else:
            data = pd.DataFrame(data=results)
        data.to_csv(os.path.join(self.save_path, self.exp_name, f'ep_log_{mode}.csv'), index=False)
