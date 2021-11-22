### YOUR CODE HERE
# import tensorflow as tf
# import torch
import os, time
import numpy as np
import torch
from Network import MyNetwork
from torch import nn
from tqdm import tqdm
from Utils import get_most_recent_ckpt_path, plot_results

"""This script defines the training, validation and testing process.
"""

class MyModel(nn.Module):

    def __init__(self, configs, max_lr=1e-2, loss_func=nn.functional.cross_entropy, optim=torch.optim.Adam, load_checkpoint_num=0, model_name="model"):
        super(MyModel, self).__init__()
        # self.configs = configs
        self.network = MyNetwork(in_channels= 3, num_classes= 10)
        if(load_checkpoint_num!=0): self.network=self.network.load(load_checkpoint_num)
        self.load_checkpoint_num=load_checkpoint_num
        self.max_lr=max_lr
        self.loss_func=loss_func
        self.optim=optim
        self.dir_path="outputs_"+model_name+"/"
        if not os.path.exists(self.dir_path): os.makedirs(self.dir_path)

    def train_validate(self, epochs, train_dataset_loaded, valid_dataset_loaded):
        optimizer = self.optim(self.network.parameters(), self.max_lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.max_lr, epochs * len(train_dataset_loaded))
        
        results = []
        for epoch in range(epochs):
            self.network.train()
            train_losses = []
            lrs = []
            for images, labels in tqdm(train_dataset_loaded):
                logits = self.network(images)
                loss = self.loss_func(logits, labels)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                lrs.append(optimizer.param_groups[0]['lr'])
                scheduler.step()
                # self.batch_iter=self.batch_iter+1
                # print("Completed batch_iter: ", self.batch_iter)
                break
            epoch_train_loss = torch.stack(train_losses).mean().item()
                        
            epoch_avg_loss, epoch_avg_acc = self.evaluate(valid_dataset_loaded, test=False)

            results.append({'avg_valid_loss': epoch_avg_loss, "avg_valid_acc": epoch_avg_acc, "avg_train_loss" : epoch_train_loss, "lrs" : lrs})

        self.network.save(self.dir_path, self.load_checkpoint_num+epochs)
        np.save(self.dir_path+"training_results", np.array(results), allow_pickle=True)
        plot_results(results, self.dir_path)
        return results

    
    def accuracy(self, logits, labels):
        pred, predClassId = torch.max(logits, dim = 1) 
        return torch.tensor(torch.sum(predClassId == labels).item()/ len(logits)*100)
    
    def evaluate(self, dataset_loaded, test=True):

        if(test):
            ckpt_path=get_most_recent_ckpt_path(self.dir_path)
            self.network.load_ckpt(ckpt_path)

        self.network.eval()
        batch_losses, batch_accs = [], []                   
        for images, labels in dataset_loaded:
            with torch.no_grad():
                logits = self.network(images)
            batch_losses.append(self.loss_func(logits, labels))
            batch_accs.append(self.accuracy(logits, labels))
        avg_loss = torch.stack(batch_losses).mean().item()
        avg_acc = torch.stack(batch_accs).mean().item()
        return avg_loss, avg_acc

    def predict_prob(self, test_dataset_loaded):
        results = self.evaluate(test_dataset_loaded)


### END CODE HERE