from models.abstract_architecture import AbstractArchitecture
import torch.nn as nn


class AE(AbstractArchitecture):
    """
    Auto Encoder network implemented as a two part design with encoder and
    decoder part. Using softplus for hidden and tanh for output layers.
    """
    def __init__(self):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Softplus(),
            nn.Linear(256, 128),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.Softplus(),
            nn.Linear(256, 1024),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        x = self.encoder(x)
        return x

    def decode(self, x):
        return self.decoder(x)

    def create_output(self, item):
        return self.encode(item)


class AEEmbedding(AbstractArchitecture):
    """
    Auto Encoder for sequence data optimizing the letter encoding
    """
    def __init__(self):
        super(AEEmbedding, self).__init__()
        self.hidden_size = 10
        self.kernel_size = 4
        self.stride = 4

        self.emb = nn.Embedding(130, self.hidden_size)
        self.conv = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=self.kernel_size, stride=self.stride)
        self.deconv = nn.ConvTranspose1d(self.hidden_size, self.hidden_size, kernel_size=self.kernel_size,
                                         stride=self.stride)
        self.fc = nn.Linear(self.hidden_size, 130)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        x = self.emb(x)
        x = self.conv(x.transpose(1, 2))
        return x.transpose(1, 2)

    def decode(self, x):
        x = self.deconv(x.transpose(1, 2))
        x = self.fc(x.transpose(1, 2))
        return x

    def create_output(self, x):
        return self.encode(x)

