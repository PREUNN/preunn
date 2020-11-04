from data.preprocessors.image_preprocessing.processing import *
from data.preprocessors.abstract_preprocessor import *
from abc import ABC


class AbstractImagePreprocessor(AbstractPreprocessor, ABC):
    """Abstract super class for image preprocessed datasets"""

    def __init__(self, source_dataset: AbstractDataset, data_length: int):
        super().__init__(source_dataset=source_dataset)
        assert data_length > 0
        self.data_length = data_length

    def split(self, split_value: float):
        """
        Function to split the dataset inside the preprocessor and returning new preprocessors accordingly
        :param split_value: Value of how to split the data. Needs to be inside the interval [0, 1]
        :return: 2 new preprocessors of the same type with the separated parts of the dataset
        """
        # parameter checks
        assert 0 <= split_value <= 1

        # creating new preprocessors
        first_pp = self.__class__(source_dataset=self.source_dataset, data_length=self.data_length)
        second_pp = self.__class__(source_dataset=self.source_dataset, data_length=self.data_length)
        first_pp.source_dataset, second_pp.source_dataset = self.source_dataset.split(split_value=split_value)
        return first_pp, second_pp


class NormalImagePreprocessor(AbstractImagePreprocessor):
    """Unaltered dataset based on the image interpretation of bytes, uses padding"""

    def __getitem__(self, idx):
        # get base item and label (following item) from dataset
        if idx == self.source_dataset.__len__() - 1:
            return self.__getitem__(random.randint(0, idx-1))
        item = self.source_dataset.__getitem__(idx)
        label = self.source_dataset.__getitem__(idx+1)

        # padding item and label
        item = padding_string(string=item, data_length=self.data_length)
        label = padding_string(string=label, data_length=self.data_length)

        # image interpretation from item and label
        item_tensor = string_to_tensor(string=item)
        label_tensor = string_to_tensor(string=label)
        item_tensor = tensor_normalization(item_tensor, min=self.source_dataset.data_min_value,
                                           max=self.source_dataset.data_max_value)
        label_tensor = tensor_normalization(label_tensor, min=self.source_dataset.data_min_value,
                                            max=self.source_dataset.data_max_value)
        return item_tensor, label_tensor


class ScrambledImagePreprocessor(AbstractImagePreprocessor):
    """Scrambled dataset based on the image interpretation of bytes"""
    intervals = [4, 8, 16, 32, 64, 1024]

    def __init__(self, source_dataset: AbstractDataset, data_length: int):
        super().__init__(source_dataset=source_dataset, data_length=data_length)

    def __getitem__(self, idx):
        # get base item from dataset
        item = self.source_dataset.__getitem__(idx)

        # choosing a random interval from list
        i = random.randint(0, (len(self.intervals)-1))

        # creating tensor using scramble and padding methods
        item = scramble_string(string=item, length_of_piece=self.intervals[i])
        item = padding_string(string=item, data_length=self.data_length)
        item_tensor = string_to_tensor(string=item)
        item_tensor = tensor_normalization(item_tensor, min=self.source_dataset.data_min_value,
                                           max=self.source_dataset.data_max_value)

        # creating class label
        label = torch.LongTensor([i])

        # return item tensor and label as tuple
        return item_tensor, label


class FuzzyImagePreprocessor(NormalImagePreprocessor):
    """Normal image dataset but with fuzzy labels from 0.8 to 1 on the image interpretation of bytes"""

    def __init__(self, source_dataset: AbstractDataset, data_length: int):
        super().__init__(source_dataset=source_dataset, data_length=data_length)

    def __getitem__(self, idx):
        # get base item from dataset and setting a fuzzy label
        item_tensor, _ = super().__getitem__(idx)
        label = torch.FloatTensor(1).uniform_(0.8, 1.)
        return item_tensor, label


class AEImagePreprocessor(AbstractImagePreprocessor):
    """Preprocessor using an image based Auto Encoder as backbone and normal image preprocessor as base"""

    def __init__(self, source_dataset: AbstractDataset, data_length: int, autoencoder):
        super().__init__(source_dataset=source_dataset, data_length=data_length)
        self.autoencoder = autoencoder
        self.nip = NormalImagePreprocessor(self.source_dataset, self.data_length)

    def __getitem__(self, idx):
        # get item from base preprocessor and code it with the auto encoder
        item = self.nip.__getitem__(idx)
        item = (self.autoencoder.create_output(item[0]), self.autoencoder.create_output(item[1]))
        return item


class ClusteringPreprocessor(AbstractImagePreprocessor):
    """Preprocessor combining a feature extractor and SOM image backbones with a normal image preprocessor base
    and sequencing them"""

    def __init__(self, source_dataset: AbstractDataset, data_length: int, feature_extractor, som, sequence_length: int):
        super().__init__(source_dataset=source_dataset, data_length=data_length)
        self.feature_extractor = feature_extractor
        self.som = som
        self.sequence_length = sequence_length
        self.nip = NormalImagePreprocessor(self.source_dataset, self.data_length)

    def __getitem__(self, idx):
        # get first item for sequence
        idx = idx * self.sequence_length
        if self.nip.__len__() - idx <= self.sequence_length:
            return self.__getitem__(random.randint(0, idx))
        data_list = []

        # getting a sequence length of item and preparing them
        for i in range(self.sequence_length+1):
            data, _ = self.nip.__getitem__(idx + i)
            data = self.feature_extractor.create_output(data.cuda())
            data = self.som.winner(data.cpu().squeeze(0).detach())[1]
            data = torch.tensor(data)
            data_list.append(data)

        # output
        stacked_data = torch.stack(data_list)
        return stacked_data[:-1], stacked_data[1:]

    def __len__(self):
        return int(self.source_dataset.__len__() / self.sequence_length)
