### YOUR CODE HERE
# import tensorflow as tf
# import torch
import os, time
import numpy as np
import torch
from Network import MyNetwork
from torch import nn
from tqdm import tqdm
from Utils import get_most_recent_ckpt_path, get_ckpt_number, plot_results
from SummaryUtils import summary_string  #use torchsummary package from https://github.com/sksq96/pytorch-summary
from Utils import get_torch_device
import math

"""This script defines the training, validation and testing process.
"""

class MyModel(nn.Module):

    def __init__(self, configs, max_lr=1e-1, loss_func=nn.functional.cross_entropy,load_checkpoint_num=0, model_name="model"):
        super(MyModel, self).__init__()
        # self.configs = configs
        self.network = MyNetwork(depth=40, growthrate=48, in_channels= 3, num_classes= 10)
        
        if(load_checkpoint_num!=0): self.network=self.network.load(load_checkpoint_num)
        self.load_checkpoint_num=load_checkpoint_num
        self.max_lr=max_lr
        self.loss_func=loss_func
        self.optim=torch.optim.SGD
        self.dir_path="outputs_"+model_name+"/"
        self.dir_path_fin="outputs_"+model_name+"_fin/"
        if not os.path.exists(self.dir_path): os.makedirs(self.dir_path)
        if not os.path.exists(self.dir_path_fin): os.makedirs(self.dir_path_fin)

    def train_validate(self, epochs, train_dataset_loaded, valid_dataset_loaded):
        optimizer = self.optim(params=self.network.parameters(), lr=self.max_lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[math.floor(epochs/2), math.floor(3*epochs/4)], gamma=0.1)
        
        results = []
        ckpt_path=get_most_recent_ckpt_path(self.dir_path_fin)
        if ckpt_path is not None:
            self.network.load_ckpt(ckpt_path)
            self.load_checkpoint_num=get_ckpt_number(ckpt_path)

        for epoch in range(self.load_checkpoint_num+1):
            scheduler.step()

        for epoch in range(self.load_checkpoint_num+1, epochs):
            self.network.train()
            train_losses = []
            lrs = []
            print("epoch: ", epoch)
            for images, labels in tqdm(train_dataset_loaded):
                logits = self.network(images)
                loss = self.loss_func(logits, labels)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
            epoch_train_loss = torch.stack(train_losses).mean().item()
                        
            epoch_avg_loss, epoch_avg_acc = self.evaluate(valid_dataset_loaded, test=False)

            results.append({'avg_valid_loss': epoch_avg_loss, "avg_valid_acc": epoch_avg_acc, "avg_train_loss" : epoch_train_loss, "lrs" : lrs})

            if ((epoch)%20==0):
                checkpoint_num=epoch
                self.network.save(self.dir_path, checkpoint_num)
                np.save(self.dir_path+"training_results_"+str(checkpoint_num), np.array(results), allow_pickle=True)

        self.network.save(self.dir_path_fin, epochs)
        np.save(self.dir_path_fin+"training_results_fin", np.array(results), allow_pickle=True)
        plot_results(results, self.dir_path+"training_results_fin/")

        with open(self.dir_path_fin+'model_summary.log', 'w') as f:
            report, _ = summary_string(self.network, input_size=(3, 32, 32), device=get_torch_device())
            f.write(report)
            f.close
        
        return results

    
    def accuracy(self, logits, labels):
        pred, predClassId = torch.max(logits, dim = 1) 
        return torch.tensor(torch.sum(predClassId == labels).item()/ len(logits)*100)
    
    def evaluate(self, dataset_loaded, test=True):

        if(test):
            ckpt_path=get_most_recent_ckpt_path(self.dir_path_fin)
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

        if (test):
            print("avg_loss, ", avg_loss)
        print("avg_acc, ", avg_acc)
        return avg_loss, avg_acc

    def predict_prob(self, test_dataset_loaded, private=False):
        ckpt_path=get_most_recent_ckpt_path(self.dir_path_fin)
        self.network.load_ckpt(ckpt_path)
        self.network.eval()
        logits=None
        if private:
            for images in test_dataset_loaded:
                with torch.no_grad():
                    if logits is None:
                        logits = self.network(images).cpu().numpy()
                    else: logits=np.vstack((logits, self.network(images).cpu().numpy()))
        else:
            for images,_ in test_dataset_loaded:
                if logits is None:
                    logits = self.network(images).numpy()
                else: logits=np.vstack((logits, self.network(images).numpy()))

        probab_logits=nn.functional.softmax(torch.FloatTensor(logits), dim=1)
        _, predicted_class = torch.max(probab_logits, dim = 1)
        
        np.save(self.dir_path+"private_probabilities",  probab_logits)
        np.save(self.dir_path+"private_predicted_class",  predicted_class.numpy())

        return probab_logits


### END CODE HERE