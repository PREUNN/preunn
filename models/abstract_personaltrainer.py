from models.abstract_architecture import *
from abc import ABC, abstractmethod
from torch.utils.data.dataloader import DataLoader
import torch
import matplotlib.pyplot as plt


class AbstractPersonalTrainer(ABC):
    """
    a class meant to be the abstract version of other tester classes
    """

    def __init__(self, model: AbstractArchitecture, training_data: DataLoader, test_data: DataLoader, log_interval: int,
                 model_save_path: str, criterion, optimizer):
        """
        standard init method
        :param model: instance of the architecture to be trained
        :param training_data: dataloader of the training set
        :param test_data: dataloader of the test set
        :param log_interval: how many batches to train before printing loss
        :param model_save_path: path where the model is to be saved
        :param criterion: loss function for the training
        :param optimizer: optimizer for training
        """
        self.model = model
        self.training_data = training_data
        self.test_data = test_data
        self.log_interval = log_interval
        self.model_save_path = model_save_path
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            print(torch.cuda.device_count(), " GPUs are being used for training")
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(device=self.device)
        self.fig1 = None
        self.training_loss_curve = []
        self.test_loss_curve = []
        self.initialize_imaging()
        return

    def initialize_imaging(self):
        self.fig1, (self.ax1, self.ax2) = plt.subplots(2, sharey=True)
        self.ax1.set(xlabel='Iterations', ylabel='Loss')
        self.ax1.set_title("Training Loss")
        self.ax2.set(xlabel='Iterations', ylabel='Loss')
        self.ax2.set_title("Validation Loss")
        self.training_loss_curve = []
        self.test_loss_curve = []
        return

    def run_training(self, num_epochs: int):
        """
        training process
        :return: None
        """
        for epoch in range(num_epochs):
            print("epoch: ", epoch)
            self.train(epoch)
            self.test(epoch)
            maximum = self.get_loss_maximum()
            self.ax1.plot(self.training_loss_curve, "tab:blue")
            self.ax1.set_ylim(bottom=0 - maximum / 20, top=maximum*1.05)
            self.ax2.plot(self.test_loss_curve, "tab:orange")
            self.ax2.set_ylim(bottom=0 - maximum / 20, top=maximum*1.05)
            self.model.store_model(self.model_save_path)
            self.fig1.show()
        return

    def get_loss_maximum(self):
        try:
            train_maximum = max(self.training_loss_curve)
        except:
            train_maximum = 1
        try:
            test_maximum = max(self.test_loss_curve)
        except:
            test_maximum = 1
        return max(train_maximum, test_maximum)

    @abstractmethod
    def train(self, epoch: int):
        pass

    @abstractmethod
    def test(self, epoch: int):
        pass

    def print_training_loss(self, epoch: int, batch_id: int, batch_size: int, loss: int):
        print("Training Epoch: {}, [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch, batch_id * batch_size, len(self.training_data.dataset),
                                                                           100. * batch_id * batch_size / len(self.training_data.dataset), loss))
        self.training_loss_curve.append(loss)
        return

    def print_test_loss(self, epoch: int, batch_id: int, batch_size: int,  loss: int):
        print("Test Epoch: {}, [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch, batch_id * batch_size, len(self.test_data.dataset),
                                                                       100. * batch_id * batch_size / len(self.test_data.dataset), loss))
        self.test_loss_curve.append(loss)
        return

    def set_testset(self, dataloader: DataLoader):
        """
        Setter for new test dataset
        :param dataloader: Dataloader with new dataset
        :return: None
        """
        self.test_data = dataloader
        return

    def finalize_test(self):
        """
        Resetting graphs and retesting architecture
        :return: None
        """
        self.initialize_imaging()
        self.test(1)
        maximum = self.get_loss_maximum()
        self.ax2.set_title("Test Loss")
        self.ax2.plot(self.test_loss_curve, "tab:orange")
        self.ax2.set_ylim(bottom=0 - maximum / 20, top=maximum*1.05)
        self.fig1.delaxes(self.ax1)
        self.fig1.show()
        return
