import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

"""This script implements the functions for data augmentation
and preprocessing.
"""

def parse_record(record, training):
    """Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [32, 32, 3].
    """
    ### YOUR CODE HERE
    
    # Reshape from [depth * height * width] to [depth, height, width].
    depth_major = record.reshape((3, 32, 32))

    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(depth_major, [1, 2, 0])

    image = preprocess_image(image, training)

    # Convert from [height, width, depth] to [depth, height, width]
    image = np.transpose(image, [2, 0, 1])

    ### END CODE HERE

    image = preprocess_image(image, training) # If any.

    return image


def preprocess_image(image, training):
    """Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [32, 32, 3]. The processed image.
    """
    ### YOUR CODE HERE
    #Reference for stats values: https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
    cifar_mean_stddev = ((0.49139968,  0.48215841,  0.44653091), (0.24703223,  0.24348513,  0.26158784))
    #Tutorial for using pytorch transforms:https://pytorch.org/vision/stable/auto_examples/plot_scripted_tensor_transforms.html
    preprocess_train = nn.Sequential([
        transforms.RandomCrop(32,padding = 4, padding_mode = 'reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*cifar_mean_stddev, inplace=True) 
    ])
    
    preprocess_test = nn.Sequential([
         transforms.ToTensor(),
        transforms.Normalize(*cifar_mean_stddev, inplace=True)
    ])

    ### END CODE HERE

    return image