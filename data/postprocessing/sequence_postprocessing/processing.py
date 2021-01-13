import torch

# constants
ASCII_SIZE = 128


def letter_tensor_to_char(tensor, coded=False, num_clusters=1):
    """
    Turns a single one-hot-encoded tensor (for a char) back into a char. Returns
    special strings at EOP/SOP. Assuming 128 character ASCII table.
    :param tensor: One hot encoded tensor
    :param coded: If the tensor is using indexed (=False) or one hot encoding (=True)
    :param num_clusters: Number of clusters to be assumed
    :return: Char of the tensor or None if outside of ascii
    """
    # parameter checks
    assert 0 < num_clusters

    # go through all the cases
    if coded:
        _, i = tensor.data.topk(1)
    else:
        i, _ = tensor.data.topk(1)
    if ASCII_SIZE <= i < ASCII_SIZE + num_clusters:
        x = str(i.item() - ASCII_SIZE)
        return "SOP°" + x + "\n"
    if ASCII_SIZE + num_clusters <= i < ASCII_SIZE + 2 * num_clusters:
        x = str(i.item() - (ASCII_SIZE + num_clusters))
        return "EOP°" + x + "\n\n"
    else:
        return chr(i)


def char_to_letter_tensor(char: str, alphabet_size: int):
    """
    Method to one-hot encode a char to a given alphabet size
    :param char: Char to transform
    :param alphabet_size: Size of the alphabet and thus the vector
    :return: New tensor with one-hot encoding
    """
    # parameter checks
    assert len(char) == 1
    assert alphabet_size >= 128

    # one-hot encoding
    letter_tensor = torch.zeros([1, 1, alphabet_size])
    letter_tensor[0][0][ord(char)] = 1
    return letter_tensor


def print_sample(sample, coded=False):
    """
    Printing a one-hot encoded string tensor back into a char. Tensor needs to
    be 1 dimensional.
    :param sample: Sample tensor to print
    :param coded: If the tensor is using indexed (=False) or one hot encoding
    (=True)
    :return: None
    """
    # parameter checks
    assert sample
    assert len(sample.shape) == 1

    string = ""
    for letter in sample:
        string += letter_tensor_to_char(letter, coded)
    print(string)


def sequence_tensor_to_string_list(tensor, num_classes: int = 1, coded=False):
    """
    Turns a sequence tensor (one-hot encoded) into a list of strings.
    :param tensor: Tensor of shape "batchsize * sequence length * coding"
    :param coded: If the tensor is using indexed (=False) or one hot encoding (=True)
    :param num_classes: Number of classes.
    :return: List of strings from the tensor
    """
    # parameter checks
    if coded:
        assert len(tensor.shape) == 3
    else:
        assert len(tensor.shape) == 2

    string_list = []
    # iterating over all lines in the tensor
    for lines in tensor:
        string = ""

        # iterating over letters in lines
        for letter in lines:
            string += letter_tensor_to_char(letter, coded, num_clusters=num_classes)

        # output
        string_list.append(string)
    return string_list
