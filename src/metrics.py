from torchmetrics.classification import MulticlassAccuracy


class LossLog:
    def __init__(self):
        self.losses = {'train': [], 'test': []}

    def update_loss(self, new_loss, train):
        self.losses[train].append(new_loss)

    def compute_average_loss(self, train):
        return sum(self.losses[train]) / len(self.losses[train])

    def reset(self):
        self.losses = {'train': [], 'test': []}


class AccLog:
    def __init__(self, classes: int = 9, device: str = 'cuda'):
        self.accuracies = {'train': MulticlassAccuracy(classes, average='macro').to(device),
                           'test': MulticlassAccuracy(classes, average='macro').to(device)}

    def update_acc(self, preds, target, train):
        self.accuracies[train].update(preds, target)

    def compute_average_acc(self, train):
        return self.accuracies[train].compute()

    def reset(self):
        self.accuracies['train'].reset()
        self.accuracies['test'].reset()
