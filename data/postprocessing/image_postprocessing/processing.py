import numpy
import torch


def image_tensor_to_string_list(tensor):
    """
    Method to return a normalized image tensor back to byte/letter form
    :param tensor: 1D-image tensor of shape "batchsize * 1 * features"
    :return: List of strings from the tensor
    """
    # constants
    MAX = 127
    MIN = 0

    string_list = []

    # copy with detach for use in mid of training
    copy = torch.clone(tensor).detach_()
    copy = copy.cpu()

    # iterating over tensor elements
    for row in copy:
        chars = []

        # iterating over characters
        for c in row.squeeze_():
            c = (c / 2) * (MAX - MIN)   # for [-1, 1] interval
            c += ((MIN+MAX)/2)
            c = numpy.maximum(0, c)
            c = numpy.minimum(c, 127)
            chars.append(chr(int(c)))

        # output
        string_list.append("".join(chars))
    return string_list




