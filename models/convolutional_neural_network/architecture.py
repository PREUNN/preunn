from models.abstract_architecture import AbstractArchitecture
import torch.nn as nn


class CNN(AbstractArchitecture):
    """
    CNN Network using scrambled data for supervised learning
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.dropout = 0.5

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=4, stride=2),
            nn.BatchNorm1d(128),
            nn.Softplus(),
            nn.Dropout(self.dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=4, stride=2),
            nn.BatchNorm1d(64),
            nn.Softplus(),
            nn.Dropout(self.dropout),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=4, stride=2),
            nn.BatchNorm1d(32),
            nn.Softplus(),
            nn.Dropout(self.dropout),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=4, stride=2),
            nn.BatchNorm1d(16),
            nn.Softplus(),
            nn.Dropout(self.dropout),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(16, 8, kernel_size=4, stride=2),
            nn.BatchNorm1d(8),
            nn.Softplus(),
            nn.Dropout(self.dropout),
        )
        self.out = nn.Sequential(
            nn.Linear(240, 40),
            nn.Softplus(),
            nn.Linear(40, 6),
        )

    def forward(self, x):
        x = self.create_output(x)
        x = self.out(x)
        return x

    def get_feature_maps(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def create_output(self, x):
        x = self.get_feature_maps(x)
        x = x.view(x.shape[0], x.shape[1] * x.shape[2])
        return x
