import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import random
import matplotlib.pyplot as plt
#import albumentations as A
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize
import os
import matplotlib.pyplot as plt

## Model architecture

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.relu = nn.ReLU()
        
        self.pool = nn.MaxPool3d(3,2)
        
        self.conv1 = nn.Conv3d(in_channels = 2, 
                               out_channels = 96, 
                               kernel_size = 11, stride = (1,5,5))
        
        self.conv2 = nn.Conv3d(in_channels = 96, 
                               out_channels = 256, 
                               kernel_size = 5, stride = 1, padding=(2,2,2))
        
        self.conv3 = nn.Conv3d(in_channels = 256, 
                               out_channels = 384, 
                               kernel_size = 3, stride = 1,padding=(1,1,1))
        
        self.conv4 = nn.Conv3d(in_channels = 384, 
                               out_channels = 384, 
                               kernel_size = 3, stride = 1,padding=(1,1,1))
        
        self.conv5 = nn.Conv3d(in_channels = 384, 
                               out_channels = 256, 
                               kernel_size = 3, stride = 1,padding=(1,1,1))
        
        self.dropout = nn.Dropout(0.5)
        
        self.lin1 = nn.Linear(19200, 400)
        
        self.lin2 = nn.Linear(400, 400)
        
        self.lin3 = nn.Linear(400, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(len(x),-1)
        x = self.dropout(x)
        x = self.lin1(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = self.lin3(x)
        
        return x
        
        

class image_dataset(Dataset):
    def __init__(self, data,transform = False):
        self.image_path = data[0]
        self.mask_path = data[1]
        self.labels = data[2]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        self.single_image  = np.load(self.image_path[idx])
        self.single_mask = np.load(self.mask_path[idx])
        self.single_label = self.labels[idx]
        self.resized_single_image = resize(self.single_image,(48,256,256))
        self.resized_single_mask = resize(self.single_mask,(48,256,256))
        self.image_n_mask = np.stack((self.resized_single_image,self.resized_single_mask), axis = 0)
        return (self.image_n_mask, self.single_label)

device = torch.device('cuda:0')
### Reading image files
mask_vol_all = pd.read_csv('../zhout/Documents/mask_vol_all.csv')

### Sampling 200 postive and negative points

pos_samples = mask_vol_all[mask_vol_all['Failure-binary'] == 1][:200]
neg_samples = mask_vol_all[mask_vol_all['Failure-binary'] == 0][:200]

mask_vol_all = pd.concat([pos_samples, neg_samples], axis = 0 )

msk = np.random.rand(len(mask_vol_all)) < 0.8
train_mask_vol_all = mask_vol_all[msk].reset_index()
valid_mask_vol_all = mask_vol_all[~msk].reset_index()

train_path  = (list(train_mask_vol_all['image_path']), list(train_mask_vol_all['mask_path']), list(train_mask_vol_all['Failure-binary']))
valid_path  = (list(valid_mask_vol_all['image_path']), list(valid_mask_vol_all['mask_path']), list(valid_mask_vol_all['Failure-binary']))


all_data = (list(mask_vol_all['image_path']), list(mask_vol_all['mask_path']), list(mask_vol_all['Failure-binary']))


### Creating image dataset

train_data = image_dataset(train_path)
valid_data = image_dataset(valid_path)

### Creating Dataloaders

train_dl = DataLoader(train_data, batch_size = 5, shuffle = True)
valid_dl = DataLoader(valid_data, batch_size = 5, shuffle = True)

### Moving Model to GPU

model = AlexNet()
model = model.to(device)

def plot_loss(train_loss, valid_loss):
    plt.title('loss over epochs')
    ax = plt.subplot(111)
    epochs = [i+1 for i in range(len(train_loss))]
    ax.plot(epochs, train_loss, label = 'train_loss')
    ax.plot(epochs, valid_loss, label = 'valid_loss')
    ax.set_ylim([0, 1])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend()
    plt.savefig('loss_plot.png')

def model_loss(model,dl):
    print('loss_time')
    y_preds = []
    ys = []
    loss_arr = []
    model.eval()
    for x,y in dl:
        with torch.no_grad():
            x,y = x.to(device), y.to(device)
            y_hat = model(x.float())
            loss = F.binary_cross_entropy_with_logits(y_hat.squeeze(1),y.float())
            loss_arr.append(loss.cpu())
            y_preds.append(y_hat.cpu().numpy())
            ys.extend(y.cpu().numpy())
        del x,y
    y_preds_ = np.array([1. if y >=0.5 else 0. for y in np.concatenate(y_preds)])
    correct = (ys == y_preds_).sum()
    accuracy = correct/len(ys)    
    return np.mean(loss_arr),accuracy        


def training_loop(epochs, model, optimizer, train_dl, valid_dl):
    train_loss_array = []
    valid_loss_array = []
    for i in range(epochs):
        count = 0 
        print(f"###### epoch {i+1} ######")    
        model.train()
        for x,y in train_dl:
            count +=1
            x,y = x.to(device), y.to(device)
            y_hat = model(x.float())
            loss = F.binary_cross_entropy_with_logits(y_hat.squeeze(1),y.float())
            optimizer.zero_grad()
            loss.backward()
            if count % 10 == 0 :
                print(loss)
            optimizer.step()
            del x,y    
        model.eval() 
        train_loss,train_accuracy = model_loss(model,train_dl)
        valid_loss,valid_accuracy = model_loss(model,valid_dl)
        train_loss_array.append(train_loss)
        valid_loss_array.append(valid_loss)
        if valid_loss < best_loss:
            torch.save(model, '/home/mukherjeea/Alex_net_trained')
        print(f" train_loss {train_loss} valid_loss {valid_loss}")
    model_loss(train_loss_array, valid_loss_array)
    return model

### Using learning rate 0.1 and wd 0    

lr = .01
wd = 0
optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = wd)
train_loss_array,valid_loss_array,model = training_loop(100,model, optimizer, train_dl, valid_dl)

### Saving model    

