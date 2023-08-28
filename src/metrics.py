class LossLog:
    def __init__(self):
        self.losses = {'train': [], 'test': []}

    def update_loss(self, new_loss, train):
        self.losses[train].extend(new_loss)

    def compute_average_loss(self, train):
        return sum(self.losses[train]) / len(self.losses[train])

    def reset(self):
        self.losses = {'train': [], 'test': []}


class AccLog:
    def __init__(self):
        self.accuracies = {'train': [], 'test': []}

    def update_acc(self, new_acc, train):
        self.accuracies[train].extend(new_acc)

    def compute_average_acc(self, train):
        return sum(self.accuracies[train]) / len(self.accuracies[train])

    def reset(self):
        self.accuracies = {'train': [], 'test': []}
