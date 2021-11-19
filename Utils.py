import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

class Private_Image_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        load_data = np.load(data_dir+"\private_test_images_v3.npy")
        self.data=load_data.reshape(2000, 32, 32, 3)
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        x = Image.fromarray(img)
        x = self.transform(x)

        return x

    def __len__(self):
        return len(self.data)

cifar_mean_stddev = ((0.49139968,  0.48215841,  0.44653091), (0.24703223,  0.24348513,  0.26158784))

def display_batch(dataset_loaded, test=False):
    (mean, stddev)=cifar_mean_stddev
    mean = torch.tensor(mean).reshape(1,3,1,1)
    stddev = torch.tensor(stddev).reshape(1,3,1,1)
    if test:
        for images in dataset_loaded:
            fig, ax = plt.subplots(figsize = (10,10))
            images = images * stddev + mean
            ax.imshow(make_grid(images,nrow=10).permute(1,2,0))
            fig.savefig('test_batch.png', dpi=200) 
            break
    else: 
        for images, labels in dataset_loaded:
            fig, ax = plt.subplots(figsize = (10,10))
            images = images * stddev + mean
            ax.imshow(make_grid(images,10).permute(1,2,0))
            fig.savefig('batch.png', dpi=200) 
            print(labels)
            break