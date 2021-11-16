import os
import pickle
import numpy as np

"""This script implements the functions for reading data.
"""

def load_data(data_dir):
    """Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches
            are stored.

    Returns:
        x_train: An numpy array of shape [50000, 3072].
            (dtype=np.float32)
        y_train: An numpy array of shape [50000,].
            (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072].
            (dtype=np.float32)
        y_test: An numpy array of shape [10000,].
            (dtype=np.int32)
    """

    ### YOUR CODE HERE

    x_train = np.array([])
    y_train = np.array([])
    for i in range(1,6):
        batch_data = pickle.load(open(data_dir+"/cifar-10-batches-py/data_batch_"+str(i), 'rb'), encoding='bytes')
        if (not np.any(x_train)):    
            # If first batch of loading
            x_train=batch_data[b'data'].astype(np.float32)
            y_train=batch_data[b'labels']
        else:    
            x_train=np.vstack((x_train, batch_data[b'data'].astype(np.float32)))
            y_train=np.hstack((y_train, np.array(batch_data[b'labels'])))
            
    test_data=pickle.load(open(data_dir+"/test_batch", 'rb'), encoding='bytes')
    (x_test, y_test) = (test_data[b'data'].astype(np.float32), np.array(test_data[b'labels']))

    ### END CODE HERE

    return x_train, y_train, x_test, y_test


def load_testing_images(data_dir):
    """Load the images in private testing dataset.

    Args:
        data_dir: A string. The directory where the testing images
        are stored.

    Returns:
        x_test: An numpy array of shape [N, 32, 32, 3].
            (dtype=np.float32)
    """

    ### YOUR CODE HERE
    data=np.load(data_dir+"private_test_images_v3.npy")
    N=data.shape[0]
    x_test=data.reshape(N, 32, 32, 3)

    ### END CODE HERE

    return x_test


def train_valid_split(x_train, y_train, train_ratio=0.8):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        train_ratio: A float number between 0 and 1.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    
    ### YOUR CODE HERE
    split_index = train_ratio*x_train.shape
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]
    ### END CODE HERE

    return x_train_new, y_train_new, x_valid, y_valid

