from torch.autograd import Variable
import torch


def create_noise(dim):
    """
    Create random noise of correct dimensions
    :param dim: dimension that the return tensor should have
    :return: a new tensor with random values
    """
    noise = Variable(torch.randn(tuple(list(dim))))
    return noise