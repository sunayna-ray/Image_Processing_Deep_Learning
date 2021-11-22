from torch import nn
import math

#Initializes values in layers when called
def initialize_params(layers):
    not_list = False
    if not isinstance(layers,list):
        layers = [layers]
        not_list= True
        
    for module in layers:
        if isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()

        elif isinstance(module, nn.Conv2d):
            filter = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            #initialise all weights with normal distribution of mean =0, std dev= root2 by n
            module.weight.data.normal_(0, math.sqrt(2. / filter))

        elif isinstance(module, nn.Linear):
            module.bias.data.zero_()
    if not_list:
        return layers[0]
    return layers

##Functions as a convolution block
def bn_rl_conv2d_block(in_channels, out_channels,kernel_size,bias=False,pool=False, padding=0,eps=1e-5, momentum=0.997):
    layers_conv2d = [nn.BatchNorm2d(in_channels, eps, momentum),nn.ReLU(inplace = True)]
    layers_conv2d.append(nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = padding,bias=bias))
    if pool:
        layers_conv2d.append(nn.AvgPool2d(kernel_size=2, stride=2))        
    return initialize_params(layers_conv2d)

def bottleneck_block(nchannels,growthrate):
    intermediate_channels = 4*growthrate
    
    layers = bn_rl_conv2d_block(in_channels=nchannels,
                                out_channels=intermediate_channels,
                                kernel_size=1,bias=False)
    layers.extend(bn_rl_conv2d_block(in_channels=intermediate_channels,out_channels=growthrate,
                                        kernel_size=3,bias=False,padding=1))
    return nn.Sequential(*layers)