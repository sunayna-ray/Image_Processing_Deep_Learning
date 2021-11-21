from torch import nn
from torch.functional import Tensor

##Functions as a convolution block
def conv2_BNRL(in_channels, out_channels, eps=1e-5, momentum=0.997, pool=False):
    layers_conv2d = [nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1), 
            nn.BatchNorm2d(out_channels, eps, momentum),
            nn.ReLU(inplace = True)]
    if pool:
        layers_conv2d.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers_conv2d)