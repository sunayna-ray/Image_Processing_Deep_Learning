from Utils import Private_Image_Dataset, WrappedDataLoader, display_batch
from ImageUtils import preprocess_image
import torchvision
import torch
from torch.utils.data import DataLoader

"""This script implements the functions for reading data.
"""
batch_size=64

def load_data(data_dir=''):
    """Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches
            are stored/will be downloaded.

    Returns:
        train_dataset_loaded: Training Dataset
        valid_dataset_loaded: Validation Dataset
        cifar_test_dataset_loaded: CIFAR-10 Test Dataset
    """
    
    input_train_dataset = torchvision.datasets.CIFAR10(root = data_dir, download = True, transform = preprocess_image()[0])
    cifar_test_dataset = torchvision.datasets.CIFAR10(root = data_dir, download = True, transform = preprocess_image()[1], train = False)
    
    train_dataset, valid_dataset = train_valid_split(input_train_dataset)
    train_dataset_loaded = DataLoader(train_dataset, batch_size, shuffle = True, pin_memory = True)
    valid_dataset_loaded = DataLoader(valid_dataset, batch_size, shuffle = True, pin_memory = True)
    cifar_test_dataset_loaded = DataLoader(cifar_test_dataset, batch_size, shuffle = True, pin_memory = True)
    
    train_dataset_loaded = WrappedDataLoader(train_dataset_loaded)
    valid_dataset_loaded = WrappedDataLoader(valid_dataset_loaded)
    cifar_test_dataset_loaded = WrappedDataLoader(cifar_test_dataset_loaded)
    display_batch(cifar_test_dataset_loaded)
    ### END CODE HERE

    return train_dataset_loaded, valid_dataset_loaded, cifar_test_dataset_loaded

def load_private_testing_images(data_dir):
    """Load the images in private testing dataset.

    Args:
        data_dir: A string. The directory where the testing images
        are stored.

    Returns:
        private_test_dataset_loaded: Private Test dataset
    """

    ### YOUR CODE HERE
    private_test_dataset=Private_Image_Dataset(data_dir, transform=preprocess_image()[1])
    private_test_dataset_loaded = DataLoader(private_test_dataset, batch_size, shuffle = False, pin_memory = True, num_workers=2)
    private_test_dataset_loaded = WrappedDataLoader(private_test_dataset_loaded)
    display_batch(private_test_dataset_loaded, test=True)
    ### END CODE HERE

    return private_test_dataset_loaded

def train_valid_split(input_train_dataset, train_ratio=0.8):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        input_train_dataset: Input training dataset
        train_ratio: A float number between 0 and 1.

    Returns:
        train_dataset: Training dataset of length train_ratio*x_train.shape.
        valid_dataset: Validation dataset of length [1-train_ratio]*x_train.shape.
    """
    
    ### YOUR CODE HERE
    split_index=int(train_ratio* len(input_train_dataset))
    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset = input_train_dataset, 
        lengths = [split_index, len(input_train_dataset)-split_index],
        generator = torch.Generator().manual_seed(42))
    ### END CODE HERE

    return train_dataset, valid_dataset