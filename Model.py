### YOUR CODE HERE
# import tensorflow as tf
# import torch
import os, time
import numpy as np
import torch
from Network import MyNetwork
from torch import nn
from tqdm import tqdm

"""This script defines the training, validation and testing process.
"""

class MyModel(nn.Module):

    def __init__(self, max_lr=1e-2, loss_func=nn.functional.cross_entropy, optim=torch.optim.Adam):
        super(MyModel, self).__init__()
        # self.configs = configs
        self.network = MyNetwork(in_channels= 3, num_classes= 10)
        self.max_lr=max_lr
        self.loss_func=loss_func
        self.optim=optim
        self.batch_iter=0

    def accuracy(self, logits, labels):
        pred, predClassId = torch.max(logits, dim = 1) 
        return torch.tensor(torch.sum(predClassId == labels).item()/ len(logits))

    def train_validate(self, epochs, train_dataset_loaded, valid_dataset_loaded):
        optimizer = self.optim(self.network.parameters(), self.max_lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.max_lr, self.epochs * len(train_dataset_loaded))
        
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
                break
                # self.batch_iter=self.batch_iter+1
                # print("Completed batch_iter: ", self.batch_iter)
            epoch_train_loss = torch.stack(train_losses).mean().item()
                        
            self.network.eval()
            batch_losses, batch_accs = [], []               
            for images, labels in valid_dataset_loaded:
                with torch.no_grad():
                    logits = self.network(images)
                batch_losses.append(self.loss_func(logits, labels))
                batch_accs.append(self.accuracy(logits, labels))
            # print("batch_vld_acc", self.accuracy(logits, labels))
            epoch_avg_loss, epoch_avg_acc = self.evaluate(valid_dataset_loaded)
            results.append({'avg_valid_loss': epoch_avg_loss, "avg_valid_acc": epoch_avg_acc, "avg_train_loss" : epoch_train_loss, "lrs" : lrs})
        return results

    def evaluate(self, dataset_loaded):
        self.network.eval()
        batch_losses, batch_accs = [], []                   
        for images, labels in dataset_loaded:
            with torch.no_grad():
                logits = self.network(images)
            batch_losses.append(self.loss_func(logits, labels))
            batch_accs.append(self.accuracy(logits, labels))
        epoch_avg_loss = torch.stack(batch_losses).mean().item()
        epoch_avg_acc = torch.stack(batch_accs).mean()
        return epoch_avg_loss, epoch_avg_acc

    def predict_prob(self, test_dataset_loaded):
        self.network.eval()
        results = []
        batch_losses, batch_accs = [], []               
        for images, labels in test_dataset_loaded:
            with torch.no_grad():
                logits = self.network(images)
            batch_losses.append(self.loss_func(logits, labels))
            batch_accs.append(self.accuracy(logits, labels))
        print("batch_test_acc", self.accuracy(logits, labels))
        epoch_avg_loss, epoch_avg_acc = self.evaluate(self.network, test_dataset_loaded, self.loss_func)
        results.append({'avg_valid_loss': epoch_avg_loss, "avg_valid_acc": epoch_avg_acc})


### END CODE HERE