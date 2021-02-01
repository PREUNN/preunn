from models.abstract_architecture import AbstractArchitecture
import torch.nn as nn
from abc import ABC


class AbstractLSTMNetwork(AbstractArchitecture, ABC):
    def __init__(self, input_size: int, hidden_size: int, layers: int):
        super(AbstractLSTMNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = layers

        self.emb = nn.Embedding(self.input_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.layers)
        self.fc = nn.Linear(self.hidden_size, self.input_size)
        self.softmax = nn.Softmax(2)
        self.out = nn.Linear(self.input_size, 1)

    def forward(self, x, hc):
        x = self.emb(x)
        x, hc = self.lstm(x.transpose(0, 1), hc)
        x = self.fc(x.transpose(0, 1))
        x = self.out(x).squeeze_(2)
        return x, hc

    def init_hidden(self, batch_size: int):
        weight = next(self.parameters()).data
        return (weight.new(self.layers, batch_size, self.hidden_size).zero_(),
                weight.new(self.layers, batch_size, self.hidden_size).zero_())

    def create_output(self, item):
        return self.forward(item, self.init_hidden(len(item)))


class AbstractCELSTMNetwork(AbstractArchitecture, ABC):
    ascii_size = 128

    def __init__(self, input_size: int, hidden_size: int, layers: int):
        super(AbstractCELSTMNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = layers

        # encoder
        self.emb = nn.Embedding(self.input_size, self.hidden_size)
        self.conv = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=4, stride=4)

        # lstm
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.layers)

        # decoder
        self.deconv = nn.ConvTranspose1d(self.hidden_size, self.hidden_size, kernel_size=4, stride=4)
        self.fc = nn.Linear(self.hidden_size, self.input_size)
        self.softmax = nn.LogSoftmax(2)

    def forward(self, x, hc):
        # Encoding
        x = self.emb(x)
        x = self.conv(x.transpose(1, 2))
        x = x.transpose(1, 2)

        # LSTMing
        x, hc = self.lstm(x.transpose(0, 1), hc)
        x = x.transpose(0, 1)

        # Decoding
        x = self.deconv(x.transpose(1, 2))
        x = self.fc(x.transpose(1, 2))
        x = self.softmax(x)
        return x, hc

    def init_hidden(self, batch_size: int):
        weight = next(self.parameters()).data
        return (weight.new(self.layers, batch_size, self.hidden_size).zero_(),
                weight.new(self.layers, batch_size, self.hidden_size).zero_())

    def create_output(self, item):
        return self.forward(item, self.init_hidden(len(item)))


class LSTMNetworkFRE(AbstractCELSTMNetwork):

    def __init__(self, num_classes: int = 1):
        self.num_classes = num_classes
        super(LSTMNetworkFRE, self).__init__(input_size=self.ascii_size + 2 * self.num_classes,
                                             hidden_size=50,
                                             layers=1)


class LSTMNetworkSR(AbstractLSTMNetwork):

    def __init__(self, num_classes: int = 16, num_hidden: int = 64,
                 num_layers: int = 1):
        self.num_classes = num_classes
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        super(LSTMNetworkSR, self).__init__(input_size=self.num_classes,
                                            hidden_size=self.num_hidden,
                                            layers=self.num_layers)


class LSTMNetworkSG(AbstractCELSTMNetwork):

    def __init__(self, num_classes: int = 16):
        self.num_classes = num_classes
        super(LSTMNetworkSG, self).__init__(input_size=self.ascii_size + 2 * self.num_classes,
                                            hidden_size=100,
                                            layers=2)


