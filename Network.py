### YOUR CODE HERE
import torch
from torch import version
from torch.functional import Tensor
import torch.nn as nn
from Network_block_utils import conv2_BNRL
from collections import OrderedDict

"""This script defines the network.
"""

class MyNetwork(nn.Module):

    # def __init__(self, configs):
    def __init__(self, in_channels, num_classes=10):
        # self.configs = configs
        super().__init__()
        self.conv1 = conv2_BNRL(in_channels, 64)
        self.conv2 = conv2_BNRL(64, 128, pool = True)
        self.res1 = nn.Sequential(OrderedDict([
                ("conv1res1", conv2_BNRL(128,128)), 
                ("conv2res1", conv2_BNRL(128,128))
            ]))

        self.conv3 = conv2_BNRL(128, 256, pool = True)
        self.conv4 = conv2_BNRL(256, 512, pool = True)
        self.res2 = nn.Sequential(conv2_BNRL(512,512), conv2_BNRL(512,512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                           nn.Flatten(),
                                           nn.Dropout(0.2),
                                           nn.Linear(512, num_classes))
        self.iter=0


    '''
    Args:
        inputs: A Tensor representing a batch of input images.
        training: A boolean. Used by operations that work differently
            in training and testing phases such as batch normalization.
    Return:
        The output Tensor of the network.
    '''
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        # print("Completed Forward Pass, iter: ", self.iter)
        self.iter=self.iter +1
        return self.classifier(out)
        
    # def __call__(self, inputs, training):
    #     return self.build_network(inputs, training)

    # def build_network(self, inputs, training):
    #     return inputs

    def save(self, dir_path, epochs):
        torch.save(self.state_dict(), dir_path+'model-%d.ckpt'%(epochs))
        print("Checkpoint has been created.")

    def load_ckpt(self, ckpt_path):
      ckpt = torch.load( ckpt_path, map_location="cpu")
      self.load_state_dict(ckpt, strict=True)
      print("Restored model parameters from {}".format(ckpt_path))

    def load(self, dir_path, checkpoint_number):
      checkpoint_name='model-%d.ckpt'%(checkpoint_number)
      ckpt = torch.load( dir_path+checkpoint_name, map_location="cpu")
      self.load_state_dict(ckpt, strict=True)
      print("Restored model parameters from {}".format(checkpoint_name))

### END CODE HERE