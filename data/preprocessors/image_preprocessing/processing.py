import torch
from torch import Tensor
import numpy as np
import random


def padding_string(string: str, data_length: int, padder="0"):
    """
    Adding a padder symbol to a string to make it fit a given data length
    :param string: String to be extended
    :param data_length: Final length the string should have
    :param padder: Symbol to append to the string repeatedly
    :return: Padded string of required length
    """
    # parameter checks
    assert data_length > 0
    assert len(padder) == 1

    # length check and cut to length limit
    if len(string) > data_length:
        string = string[:data_length - 1]

    # adding missing number of padder elements
    missing = data_length - len(string)
    string += missing * padder
    return string


def string_to_tensor(string: str):
    """
    Simple routine to create a tensor from a string
    :param string: Input string to be turned into a tensor
    :return: New tensor from the ascii values of the string
    """
    # parameter checks
    assert string

    # creating tensor from string
    ordlist = [float(ord(c)) for c in string]
    arr = np.asarray(ordlist)
    arr = np.expand_dims(arr, axis=0)

    # output
    return_tensor = torch.tensor(arr, dtype=torch.float)
    return return_tensor


def tensor_normalization(tensor: Tensor, min: int, max: int):
    """
    Method to normalise the values inside an ascii tensor to the [-1, 1] interval
    :param tensor: Tensor to be normalized
    :param min: Minimum of dataset
    :param max: Maximum of dataset
    :return: Normalized dataset is returned
    """
    # parameter checks
    assert min >= 0
    assert max > min

    # normalizing to [-1, 1]
    tensor = tensor - ((min+max)/2)
    return 2*tensor / (max - min)


def scramble_string(string: str, length_of_piece: int):
    """
    This method is a core piece of preprocessing for CNNs. It will take a string and cut it into pieces of length
    length_of_piece. These pieces will then be put in a random order and returned.
    :param string: String to be scrambled
    :param length_of_piece: Assigns the length of the pieces in which the data
    will be cut
    :return: Scrambled string
    """
    # parameter checks
    assert length_of_piece > 0
    assert string

    fragment_list = []
    cut_index = 0

    # iterating over pieces in the line and cutting them
    while cut_index < len(string):
        data_fragment = string[cut_index:cut_index+length_of_piece]
        cut_index += length_of_piece
        fragment_list.append(data_fragment)

    # randomly rearrange the pieces of one data element
    random.shuffle(fragment_list)
    reformed_string = "".join(fragment_list)

    return reformed_string


def repeat_string(string: str, data_length: int):
    """
    Method to elongate a string by repeating it and cutting it off at data_length
    :param string: Base string to elongate
    :param data_length: Length the repeater_string should have on return
    :return: New string of length data_length
    """
    # parameter check
    assert data_length > 0

    # creating repeater string
    repeater_string = string
    while len(repeater_string) <= 1024:
        repeater_string += string

    return repeater_string[:1024]









