import random
import torch


def one_hot_encode_string(string):
    """
    Function to turn a string into a tensor
    :param string: Input string
    :return: New tensor of shape "len(string) * 1"
    """
    # parameter checks
    assert string

    tensor = torch.zeros(len(string)).long()

    # iterating over letters of string
    for c in range(len(string)):
        char = ord(string[c])
        try:
            if 0 <= char <= 256:
                tensor[c] = char
            else:
                tensor[c] = random.randint(0, 129)  # excluding chinese characters
        except:
            tensor[c] = random.randint(0, 129)
    return tensor
