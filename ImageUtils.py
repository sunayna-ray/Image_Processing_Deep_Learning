import numpy as np

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
    if training:
        # Resize the image to add four extra pixels on each side.
        image=np.pad(image, ((4,4), (4,4), (0,0)), 'constant')
        # Randomly crop a [32, 32] section of the image.
        (crop_size_x, crop_size_y)=(32,32)
        tl_x=np.random.randint(1, image.shape[0]-crop_size_x+2)
        tl_y=np.random.randint(1, image.shape[1]-crop_size_y+2)
        image=image[tl_x-1:tl_x+crop_size_x-1, tl_y-1:tl_y+crop_size_y-1]

        # Randomly flip the image horizontally.
        if np.random.randint(0,1) == 1: image=np.flip(image, 1)
        
    # Subtract off the mean and divide by the standard deviation of the pixels.
    image=(image-np.mean(image, axis=(0,1)))/np.std(image, axis=(0,1))
    ### END CODE HERE

    return image


# Other functions
### YOUR CODE HERE

### END CODE HERE