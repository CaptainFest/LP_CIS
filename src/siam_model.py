import torch.nn as nn
from torchvision import models


class EmbedNetwork(nn.Module):
    """
        Siamese network for image similarity estimation.
        The network is composed of two identical networks, one for each input.
        The output of each network is concatenated and passed to a linear layer.
        The output of the linear layer passed through a sigmoid function.
        `"FaceNet" <https://arxiv.org/pdf/1503.03832.pdf>`_ is a variant of the Siamese network.
        This implementation varies from FaceNet as we use the `ResNet-18` model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_ as our feature extractor.

    """

    def __init__(self, last_feat_num:int=2):
        super(EmbedNetwork, self).__init__()
        # get resnet model
        self.resnet = models.resnet18(weights=None)

        # over-write the first conv layer to be able to read MNIST images
        # as resnet18 reads (3,x,x) where 3 is RGB channels
        # self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
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


class TripletNetwork(nn.Module):

    def __init__(self, last_feat_num: int):
        super(TripletNetwork, self).__init__()
        # get resnet model
        self.embedding_net = EmbedNetwork(last_feat_num)

    def get_embedding(self, x):
        output = self.embedding_net(x)
        return output

    def forward(self, input1, input2, input3):
        output1 = self.get_embedding(input1)
        output2 = self.get_embedding(input2)
        output3 = self.get_embedding(input3)

        return output1, output2, output3




class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, emb_size: int, n_classes: int):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(emb_size, n_classes)

    def forward(self, x):
        output = self.embedding_net.get_embedding(x)
        output = self.nonlinear(output)
        scores = nn.functional.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))
