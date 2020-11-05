from data.source_datasets.datasets import AbstractDataset
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import random


class AbstractPreprocessor(Dataset, ABC):
    """Abstract super class for all preprocessors"""
    random_seed = 123456789

    def __init__(self, source_dataset: AbstractDataset):
        self.source_dataset = source_dataset

    def __len__(self):
        return self.source_dataset.__len__()

    @abstractmethod
    def __getitem__(self, idx):
        pass

    def merge(self, new_dataset: Dataset):
        """
        Merging the dataset used by this preprocessor with a new, given one
        :param new_dataset: New Dataset to add to this one
        :return: None
        """
        # parameter checks
        assert new_dataset

        self.source_dataset.merge(new_dataset)

    @abstractmethod
    def split(self, split_value: float):
        pass

    def shuffle_dataset(self):
        """
        This method will randomly shuffle the dataset used by this preprocessor
        :return:
        """
        random.Random(self.random_seed).shuffle(self.source_dataset.data)
