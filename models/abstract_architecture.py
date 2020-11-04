from abc import ABC, abstractmethod
import torch


class AbstractArchitecture(torch.nn.Module, ABC):
    """
    An abstract class for all neural networks to share code
    """
    def __init__(self):
        super(AbstractArchitecture, self).__init__()

    def store_model(self, path):
        """
        Saving the model at a given path
        :param path: where to store the model
        :return: None
        """
        torch.save(self, path)
        print("saved " + path)

    @abstractmethod
    def create_output(self, item):
        pass
