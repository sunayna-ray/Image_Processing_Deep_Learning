### YOUR CODE HERE
import torch
from torch import version
from torch.functional import Tensor
import torch.nn as nn

"""This script defines the network.
"""

class MyNetwork(nn.Module):

    def __init__(self, configs):
        self.configs = configs


    '''
    Args:
        inputs: A Tensor representing a batch of input images.
        training: A boolean. Used by operations that work differently
            in training and testing phases such as batch normalization.
    Return:
        The output Tensor of the network.
    '''
    def __call__(self, inputs, training):
        return self.build_network(inputs, training)

    def build_network(self, inputs, training):
        return inputs


### END CODE HERE