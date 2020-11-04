from models.abstract_architecture import AbstractArchitecture
import torch.nn as nn


class Discriminator(AbstractArchitecture):
    """
    A five layer discriminative neural network
    """
    def __init__(self):
        super(Discriminator, self).__init__()

        self.dropout_level = 0.2

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 10, kernel_size=4, stride=4),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout_level)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(10, 20, kernel_size=4, stride=4),
            nn.BatchNorm1d(20),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout_level)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(20, 60, kernel_size=4, stride=4),
            nn.BatchNorm1d(60),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout_level)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(60, 90, kernel_size=4, stride=4),
            nn.BatchNorm1d(90),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout_level)
        )

        self.out = nn.Sequential(
            nn.Linear(360, 1),
            nn.Sigmoid()
        )
        return

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.shape[0], 360)
        x = self.out(x)
        return x

    def create_output(self, item):
        x = self.conv1(item)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.conv4(x).view(x.shape[0], 360)


class Generator(AbstractArchitecture):
    """
    A four layer generative neural network
    """
    def __init__(self):
        super(Generator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.ConvTranspose1d(1024, 1024, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose1d(1024, 128, kernel_size=4, stride=4, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose1d(128, 32, kernel_size=8, stride=8, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose1d(32, 1, kernel_size=16, stride=16, bias=False),
        )

        self.out = nn.Sequential(
            nn.Sigmoid()
        )
        return

    def forward(self, x):
        x = x.view(x.shape[0], 1024, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

    def create_output(self, item):
        return self.forward(item)
