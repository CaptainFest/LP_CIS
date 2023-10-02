import torch
import lightning as l
import torch.nn as nn
from torchvision import models
from torch.optim import lr_scheduler, Adam


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


class LitTriplet(l.LightningModule):
    def __init__(self, last_feat_num: int, loss_fn):
        super().__init__(self)
        # get resnet model
        self.embedding_net = EmbedNetwork(last_feat_num)
        self.loss_fn = loss_fn

    def get_embedding(self, x):
        output = self.embedding_net(x)
        return output

    def forward(self, input1, input2, input3):
        output1 = self.get_embedding(input1)
        output2 = self.get_embedding(input2)
        output3 = self.get_embedding(input3)

        return output1, output2, output3

    def training_step(self, batch_data, batch_idx):
        # training_step defines the train loop. It is independent of forward
        batch_data, batch_label = batch_data
        outputs = self.embedding_net(batch_data)
        loss = self.loss_fn(outputs, batch_label)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch_data, batch_idx):
        # training_step defines the train loop. It is independent of forward
        batch_data, batch_label = batch_data
        outputs = self.embedding_net(batch_data)
        loss = self.loss_fn(outputs, batch_label)
        self.log("train_loss", loss)
        return

    def on_train_epoch_end(self):
        print(self.log)
        self.training_batch_preds.clear()

    def on_validation_epoch_end(self):
        print(self.log)
        self.validation_batch_preds.clear()

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


class LitClf(nn.Module):
    def __init__(self, embedding_net, emb_size: int, n_classes: int):
        super().__init__(self)
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(emb_size, n_classes)

        self.batch_preds = {'train': [], 'valid': []}
        self.validation_batch_preds = []

    def forward(self, x):
        output = self.embedding_net.get_embedding(x)
        output = self.nonlinear(output)
        scores = nn.functional.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))

    def _calculate_loss(self, batch, mode="train"):
        batch_data, batch_labels = batch
        outputs = self.embedding_net(batch_data)
        loss = nn.CrossEntropyLoss(outputs, batch_labels)
        batch_preds = outputs.argmax(1)
        accuracy = (batch_preds == batch_labels).sum().item() / (float(len(batch_labels) + 1e-20))
        self.batch_preds[mode].append(batch_preds)
        self.batch_labels[mode].append(batch_labels)
        self.log_dict({f"{mode}_loss": loss, "accuracy": accuracy, "progress_bar": {"accuracy": accuracy}})
        return loss, accuracy

    def training_step(self, batch_data, batch_idx):
        loss, accuracy = self._calculate_loss(batch_data, mode='train')
        self.log_dict({"train_loss": loss, "accuracy": accuracy, "progress_bar": {"accuracy": accuracy}})
        return loss

    def validation_step(self, batch_data, batch_idx):
        loss, accuracy = self._calculate_loss(batch_data, mode='train')
        self.log_dict({"train_loss": loss, "accuracy": accuracy, "progress_bar": {"accuracy": accuracy}})

    def on_train_epoch_end(self):
        all_preds = torch.hstack(self.self.batch_preds['train'])
        all_labels = torch.hstack(self.self.batch_labels['train'])
        accuracy = (all_preds == all_labels).sum().item() / (float(len(all_labels) + 1e-20))
        self.log = 1 # TO DO
        self.batch_labels['train'].clear()
        self.batch_preds['train'].clear()

    def on_validation_epoch_end(self):
        all_preds = torch.stack(self.batch_preds['valid'])
        all_labels = torch.hstack(self.self.batch_labels['valid'])
        accuracy = (all_preds == all_labels).sum().item() / (float(len(all_labels) + 1e-20))
        self.log = 1  # TO DO
        self.batch_labels['valid'].clear()
        self.batch_preds['valid'].clear()

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
