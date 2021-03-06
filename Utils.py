import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os

class Private_Image_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        load_data = np.load(data_dir+"private_test_images_v3.npy")
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
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def display_batch(dataset_loaded, test=False, private=True):
    if not os.path.exists('images/'): os.makedirs('images/')
    (mean, stddev)=cifar_mean_stddev
    mean = move_to_device(torch.tensor(mean).reshape(1,3,1,1),get_torch_device()) 
    stddev = move_to_device(torch.tensor(stddev).reshape(1,3,1,1),get_torch_device()) 
    if private:
        for images in dataset_loaded:
            fig, ax = plt.subplots(figsize = (10,10))
            images = images * stddev + mean
            ax.imshow(make_grid(images.cpu(),nrow=10).permute(1,2,0))
            fig.savefig('images/test_batch_private.png', dpi=200) 
            break
    elif test:
        for images, labels in dataset_loaded:
            fig, ax = plt.subplots(figsize = (10,10))
            images = images * stddev + mean
            ax.imshow(make_grid(images.cpu(),10).permute(1,2,0))
            fig.savefig('images/test_batch.png', dpi=200) 
            print(' '.join('%5s' % classes[labels[j]] for j in range(64)))
            break
    else: 
        for images, labels in dataset_loaded:
            fig, ax = plt.subplots(figsize = (10,10))
            images = images * stddev + mean
            ax.imshow(make_grid(images.cpu(),10).permute(1,2,0))
            fig.savefig('images/batch.png', dpi=200)
            print(' '.join('%5s' % classes[labels[j]] for j in range(64)))
            break

#Reference: https://pytorch.org/tutorials/beginner/nn_tutorial.html#wrapping-dataloader

def get_torch_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device

def move_to_device(object, device):
    if (isinstance(object, (list, tuple))):
        # tensor = tf.convert_to_tensor(array)
        return [move_to_device(obj, device) for obj in object]
    elif isinstance(object, dict):
        return [(k, move_to_device(v, device)) for k,v in object.items]
    else:
        return object.to(device=device, non_blocking = True)

class WrappedDataLoader:
    def __init__(self, dataset_loaded):
        self.dataset_loaded = dataset_loaded

    def __len__(self):
        return len(self.dataset_loaded)

    def __iter__(self):
        batches = iter(self.dataset_loaded)
        for b in batches:
            yield (move_to_device(b, device=get_torch_device()))

def get_ckpt_number(ckpt_path):
    return int(ckpt_path[ckpt_path.index('-')+1:ckpt_path.index('.')])

def get_most_recent_ckpt_path(dir_name):
    names= list(filter(lambda dir: dir.endswith("ckpt") , os.listdir(dir_name)))
    names.sort(key = lambda f: int(get_ckpt_number(f)), reverse=True)
    if(len(names)!=0):
        return dir_name+names[0]
    else: return None

def plot_results(results, path):
    plots=[{"accuracy vs epochs": ["avg_valid_acc"]}, {"Losses vs epochs" : ["avg_valid_loss", "avg_train_loss"]}, {"learning rates vs batches": ["lrs"]}]
    fig, axes = plt.subplots(len(plots), figsize = (10,10))
    for i, pair in enumerate(plots):
        for title, graphs in pair.items():
            axes[i].se_title = title
            axes[i].legend = graphs
            axes[i]
            for graph in graphs:
                axes[i].plot([result[graph] for result in results], '-x')
    plt.savefig(path+'results_plot.png')