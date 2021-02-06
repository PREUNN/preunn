from data.preprocessors.image_preprocessing.image_preprocessors \
    import NormalImagePreprocessor
from data.preprocessors.sequence_preprocessing.processing \
    import one_hot_encode_string
from data.preprocessors.abstract_preprocessor \
    import AbstractPreprocessor, AbstractDataset
from abc import ABC
import random
import torch


class AbstractSequencePreprocessor(AbstractPreprocessor, ABC):
    """Abstract super class for sequence preprocessed datasets"""

    def __init__(self, source_dataset: AbstractDataset, alphabet_size: int, data_length: int):
        super().__init__(source_dataset)
        assert alphabet_size > 0
        self.alphabet_size = alphabet_size
        self.data_length = data_length

    def split(self, split_value: float):
        """
        Function to split the dataset inside the preprocessor and returning new
        preprocessors accordingly
        :param split_value: Value of how to split the data. Needs to be inside
        the interval [0, 1]
        :return: 2 new preprocessors of the same type with the separated parts
        of the dataset
        """
        # parameter checks
        assert 0 <= split_value <= 1

        # creating new preprocessors
        first_pp = self.__class__(self.source_dataset, self.alphabet_size, self.data_length)
        second_pp = self.__class__(self.source_dataset, self.alphabet_size, self.data_length)
        first_pp.source_dataset, second_pp.source_dataset = self.source_dataset.split(split_value)
        return first_pp, second_pp


class NormalSequencePreprocessor(AbstractSequencePreprocessor):
    """
    Preprocessor returning sequence based tensors without padding. Special
    symbols will be inserted
    """

    def __init__(self, source_dataset: AbstractDataset, alphabet_size: int, data_length=None, feature_extractor=None,
                 som=None, num_clusters=1):
        super().__init__(source_dataset, alphabet_size, data_length)
        self.feature_extractor = feature_extractor
        self.som = som
        self.num_clusters = num_clusters

    def __getitem__(self, idx):
        # get item from source
        item = self.source_dataset.__getitem__(idx)
        aux_data = self.num_clusters - 1

        # backbone processing if available
        if self.feature_extractor and self.som:
            aux_data, _ = self.aux_preprocessor.__getitem__(idx)
            aux_data = self.feature_extractor.create_output(aux_data.cuda())
            aux_data = self.som.winner(aux_data.cpu().squeeze(0).detach())[1]

        # setting special symbols
        item = chr(self.alphabet_size - aux_data - 1) + item + chr(self.alphabet_size - self.num_clusters - aux_data - 1)

        # output
        item_tensor = one_hot_encode_string(item[:-1])
        target_tensor = one_hot_encode_string(item[1:])
        return item_tensor, target_tensor


class PaddedNormalSequencePreprocessor(NormalSequencePreprocessor):
    """
    Preprocessor returning padded sequence based tensors. Special symbols will
    be inserted
    """

    def __init__(self, source_dataset: AbstractDataset, alphabet_size: int, data_length: int, feature_extractor=None,
                 som=None, num_clusters=1):
        super().__init__(source_dataset, alphabet_size, data_length)
        assert data_length > 0
        self.data_length = data_length
        self.aux_preprocessor = NormalImagePreprocessor(source_dataset, data_length)
        self.feature_extractor = feature_extractor
        self.som = som
        self.num_clusters = num_clusters

    def __getitem__(self, idx):
        # get item
        item = self.source_dataset.__getitem__(idx)
        aux_data = self.num_clusters - 1

        # backbone processing if available
        if self.feature_extractor is not None and self.som is not None:
            aux_data, _ = self.aux_preprocessor.__getitem__(idx)
            aux_data = self.feature_extractor.create_output(aux_data.cuda())
            aux_data = self.som.winner(aux_data.cpu().squeeze(0).detach())[1]

        # setting special symbols
        item = chr(self.alphabet_size - aux_data - 1) + item + chr(self.alphabet_size - self.num_clusters - aux_data - 1)

        # padding
        item += (self.data_length - len(item) + 1) * "0"

        # output
        if len(item) != 1025:
            item = item[:1025]
        item_tensor = one_hot_encode_string(item[:-1])
        target_tensor = one_hot_encode_string(item[1:])
        return item_tensor, target_tensor


class RandomSequencePreprocessor(AbstractSequencePreprocessor):
    """
    Preprocessor creating random concatenated sequences from the entire dataset
    """

    SEQUENCE_LENGTH_FACTOR = 4

    def __init__(self, source_dataset: AbstractDataset, alphabet_size: int, data_length: int, feature_extractor=None,
                 som=None, num_clusters=1):
        super().__init__(source_dataset, alphabet_size, data_length)
        assert data_length > 0
        self.data_length = data_length
        self.aux_preprocessor = NormalImagePreprocessor(source_dataset, data_length)
        self.feature_extractor = feature_extractor
        self.som = som
        self.num_clusters = num_clusters

    def __getitem__(self, idx):
        # get item
        start_idx = random.randint(0, len(self.source_dataset) - 1)
        item = ""

        # create short concatenated sequence
        while len(item) < self.SEQUENCE_LENGTH_FACTOR * self.data_length:
            try:
                data = self.source_dataset.__getitem__(start_idx)
                aux_data = 0

                # backbone preprocessing if available
                if self.feature_extractor is not None and self.som is not None:
                    with torch.no_grad():
                        aux_data, _ = self.aux_preprocessor.__getitem__(start_idx)
                        aux_data = self.feature_extractor.create_output(aux_data.cuda())
                        aux_data = self.som.winner(aux_data.cpu().squeeze(1))[1]

                # setting special symbols
                item += chr(self.alphabet_size - 2 * self.num_clusters + aux_data) + data \
                        + chr(self.alphabet_size - self.num_clusters + aux_data)
                assert(all([ord(x) < self.alphabet_size for x in item]))
            except Exception as e:
                print(e)
                return self.__getitem__(random.randint(0, idx))
            start_idx += 1

        # picking random subsequence
        start_idx = random.randint(0, len(item) - (self.data_length + 1))
        end_idx = start_idx + self.data_length + 1
        seq = item[start_idx:end_idx]

        # output
        item_tensor = one_hot_encode_string(seq[:-1])
        target_tensor = one_hot_encode_string(seq[1:])
        return item_tensor, target_tensor
