### YOUR CODE HERE
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from Network_block_utils import bottleneck_block, initialize_params,bn_rl_conv2d_block
from collections import OrderedDict

"""This script defines the network.
"""
# Reference: https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf
# Reference: https://github.com/bamos/densenet.pytorch/
class MyNetwork(nn.Module):

    # def __init__(self, configs):
    def __init__(self, growthrate, depth, reduction=0.5, in_channels=3, num_classes=10):
        # self.configs = configs
        super(MyNetwork, self).__init__()
        self.growthrate=growthrate
        self.reduction=reduction

        nDenseBlocks = (depth-4) // 6
        nchannels = 2*self.growthrate
        self.conv1 = initialize_params(
            nn.Conv2d(in_channels, nchannels, kernel_size=3, padding=1, bias=False))
        
        self.dense1 = Dense_Block(nchannels, growthrate, nDenseBlocks)        
        nchannels += nDenseBlocks*growthrate
        nOutChannels = int(math.floor(nchannels*reduction))
        self.trans1 = nn.Sequential(*bn_rl_conv2d_block(in_channels=nchannels,
                                    out_channels=nOutChannels,
                                   kernel_size=1,bias=False, pool=True))
        nchannels = nOutChannels
        
        self.dense2 = Dense_Block(nchannels, growthrate, nDenseBlocks)        
        nchannels += nDenseBlocks*growthrate
        nOutChannels = int(math.floor(nchannels*reduction))
        self.trans2 = nn.Sequential(*bn_rl_conv2d_block(in_channels=nchannels,
                                    out_channels=nOutChannels,
                                   kernel_size=1,bias=False, pool=True))
        nchannels = nOutChannels

        self.dense3 = Dense_Block(nchannels, growthrate, nDenseBlocks)        
        nchannels += nDenseBlocks*growthrate

        self.classifier = initialize_params(
            nn.Sequential(nn.BatchNorm2d(nchannels),
                        nn.ReLU(),
                        nn.AvgPool2d(8),
                        nn.Flatten(),
                        nn.Linear(nchannels,num_classes),
                        nn.LogSoftmax()))
        
        self.bn1 = nn.BatchNorm2d(nchannels)
        self.fc = initialize_params(nn.Linear(nchannels,num_classes))


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
        out = self.dense1(out)
        out = self.trans1(out)
        out = self.dense2(out)
        out = self.trans2(out)
        out = self.dense3(out)
        out1 = self.classifier(out)
        out2 = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        # out2 = F.log_softmax(self.fc(out2))
        print(out2.size())
        # print(out2)
        return out1
        
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

class Dense_Block(nn.Module):
    def __init__(self, nchannels, growthrate, nDenseBlocks):
        # self.configs = configs
        super(Dense_Block, self).__init__()
        self.nchannels=nchannels
        self.growthrate=growthrate
        self.nDenseBlocks=nDenseBlocks

    def forward(self, x):
        for i in range(int(self.nDenseBlocks)):
            x = torch.cat((x,bottleneck_block(x,self.nchannels,self.growthrate) ),1)
            self.nchannels += self.growthrate
        return x


### END CODE HERE