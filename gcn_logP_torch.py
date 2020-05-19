# -*- coding: utf-8 -*-
"""
Created on Sat May  2 01:41:46 2020

@author: SFC202004009
"""


import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from utils2 import read_smiles, get_logP, convert_to_graph
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import sys
import time
import argparse

def make_partition():
    
    smi_total = np.array(read_smiles('ZINC.smiles', 50000))
    logP_total = np.array(get_logP(smi_total))
    print(smi_total.shape)
    print(logP_total.shape)
    num_train = 30000
    num_validation = 10000
    num_test = 10000
    
    smi_train = smi_total[0:num_train]
    logP_train = logP_total[0:num_train]
    smi_validation = smi_total[num_train:(num_train+num_validation)]
    logP_validation = logP_total[num_train:(num_train+num_validation)]
    smi_test = smi_total[(num_train+num_validation):]
    logP_test = logP_total[(num_train+num_validation):]
    
    train_dataset = dataSet(smi_train, logP_train)
    val_dataset = dataSet(smi_validation, logP_validation)
    test_dataset = dataSet(smi_test, logP_test)
    
    partition = {'train': train_dataset,
                 'val': val_dataset,
                 'test': test_dataset}
    
    return partition
    
class dataSet(Dataset):
    def __init__(self, smi_list, logP_list):
        self.smi_list= smi_list
        self.logP_list = logP_list
        
    def __len__(self):
        return len(self.logP_list)
    
    def __getitem__(self, idx):
        return self.smi_list[idx], self.logP_list[idx]

class Skip_conn(nn.Module):
    def __init__(self, in_dim, out_dim, act):
        super(Skip_conn, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias = False)
        
    def forward(self, in_x, out_x):
        if(self.in_dim != self.out_dim):
            in_x = self.linear(in_x)
        return nn.ReLU(in_x + out_x)

class Gated_skip(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Gated_skip, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.linear = nn.Linear(self.in_dim, self.out_dim)
        self.in_coef = nn.Linear(self.out_dim, self.out_dim)
        self.out_coef = nn.Linear(self.out_dim, self.out_dim)
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self, in_x, out_x):
        if(self.in_dim != self.out_dim):
            in_x = self.linear(in_x)
        coef = self.get_gated_coef(in_x, out_x)
        output = torch.mul(out_x, coef) + torch.mul(in_x, 1-coef)
        
        return output
    
    def get_gated_coef(self, in_x, out_x):
        x1 = self.in_coef(in_x)
        x2 = self.out_coef(out_x)
        coef = self.sigmoid(x1 + x2)
        return coef


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, sc, num_layer): #58, 64, 256, 3
        super(GCN, self).__init__()
        
        self.gnn_list = nn.ModuleList()
        
        self.gnn_list.append(nn.Linear(in_dim, hidden_dim1))
        for i in range(1, num_layer):
            gnn = nn.Linear(hidden_dim1, hidden_dim1)
            self.gnn_list.append(gnn)
        
        self.read_out = nn.Linear(hidden_dim1, hidden_dim2)
        
        self.linear1 = nn.Linear(hidden_dim2, hidden_dim2)
        self.linear2 = nn.Linear(hidden_dim2, hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2, 1)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        if(sc == 'gsc'):
            self.sc1 = Gated_skip(in_dim, hidden_dim1)
            self.sc2 = Gated_skip(hidden_dim1, hidden_dim1)
            
        elif(sc == 'skip'):
            self.sc1 = Skip_conn(in_dim, hidden_dim1)
            self.sc2 = Skip_conn(hidden_dim1, hidden_dim1)
            
        elif(sc =='no'):
            self.sc1 = nn.ReLU()
            self.sc2 = nn.ReLU()
        
        self.xavier_init()
        
    def forward(self, x, a):
        for i, gnn in enumerate(self.gnn_list):
            in_x = x
            x = torch.matmul(a, gnn(x))
            if(i == 0):
                x = self.sc1(in_x, x)
            else:
                x = self.sc2(in_x, x)
        
        x = self.read_out(x)
        x = torch.sum(x, dim = 1)
        x = self.sigmoid(x)
        
        x = self.relu( self.linear1(x) )
        x = self.tanh( self.linear2(x) )
        x = self.linear3(x)
        return x
    
    def xavier_init(self):
        for gnn in self.gnn_list:
            nn.init.xavier_normal_(gnn.weight)
            gnn.bias.data.fill_(0.01)
            
        nn.init.xavier_normal_(self.read_out.weight)
        self.read_out.bias.data.fill_(0.01)
        
        nn.init.xavier_normal_(self.linear1.weight)
        self.linear1.bias.data.fill_(0.01)
        
        nn.init.xavier_normal_(self.linear2.weight)
        self.linear2.bias.data.fill_(0.01)
        
        nn.init.xavier_normal_(self.linear3.weight)
        self.linear3.bias.data.fill_(0.01)
        

def train(net,partition, optimizer, criterion):
    dataloader = DataLoader(partition['train'], batch_size = 512, shuffle = True, num_workers = 0)
    
    net.train()
    optimizer.zero_grad()
    
    total = 0
    train_loss = 0.0
    for data in dataloader:
        inputs, labels = data
        x, a = convert_to_graph(inputs)
        # x = np.array(x)
        
        x = torch.from_numpy(x).float()
        a = torch.from_numpy(np.array(a)).float()
        x = x.cuda()
        a = a.cuda()
        labels = labels.float()
        labels = labels.cuda()
        
        outputs = net(x, a)
        
        outputs = outputs.squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        total += labels.size(0)
        
    train_loss /= len(dataloader)
    
    return net, train_loss

def validate(net, partition, criterion):
    dataloader = DataLoader(partition['val'], batch_size = 10000, shuffle = True, num_workers = 0)
    net.eval()
    
    with torch.no_grad():
        total = 0
        val_loss = 0.0
        for data in dataloader:
            inputs, labels = data
            x, a = convert_to_graph(inputs)
            # x = np.array(x)
            x = torch.from_numpy(x).float()
            a = torch.from_numpy(np.array(a)).float()
            x = x.cuda()
            a = a.cuda()
            labels = labels.float()
            labels = labels.cuda()
            
            outputs = net(x, a)
            
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_loss /= len(dataloader)
    
    return val_loss

def test(net, partition, criterion):
    dataloader = DataLoader(partition['test'], batch_size = 10000, shuffle = True, num_workers = 0)
    net.eval()
    
    with torch.no_grad():
        total = 0
        test_loss = 0.0
        for data in dataloader:
            inputs, labels = data
            x, a = convert_to_graph(inputs)
            # x = np.array(x)
            x = torch.from_numpy(x).float()
            a = torch.from_numpy(np.array(a)).float()
            x = x.cuda()
            a = a.cuda()
            labels = labels.float()
            labels = labels.cuda()
            
            outputs = net(x, a)
            
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            test_loss  += loss.item()
    test_loss  /= len(dataloader)
    
    labels = labels.cpu()
    outputs = outputs.cpu()
    
    plt.figure()
    plt.scatter(labels, outputs, s=3)
    plt.xlabel('logP - Truth', fontsize=15)
    plt.ylabel('logP - Prediction', fontsize=15)
    x = np.arange(-4,6)
    plt.plot(x,x,c='black')
    plt.tight_layout()
    plt.axis('equal')
    plt.show()
    
    return test_loss 

def experiment(args):
    net = GCN(args.in_dim, args.hidden_dim1, args.hidden_dim2, args.sc, args.num_layer)
    partition = make_partition()
    criterion = nn.MSELoss()
    
    if(args.optim == 'sgd'):
        optimizer = optim.SGD(net.parameters(), lr = args.lr, weight_decay = 1e-4)
    elif(args.optim == 'adam'):
        optimizer = optim.Adam(net.parameters(), lr = args.lr, weight_decay = 1e-4)
    
    net.cuda()
    
    epoch_size = args.epoch_size
    for i in range(epoch_size):
        for g in optimizer.param_groups:
            g['lr'] = args.lr * 0.99**i
            
        tic = time.time()
        net, train_loss = train(net, partition, optimizer, criterion)
        val_loss = validate(net, partition, criterion)
        tok = time.time()
        print("epoch: {} test loss: {:2.3f}, val_loss: {:2.3f} took: {:2.2f}".format(i, train_loss, val_loss, tok-tic))
        
    test_loss = test(net, partition, criterion)
    print("test loss: {}".format(test_loss))
    
seed = 690220955545900
seed2 = 3707523286557544
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed2)

print('torch seed: ', torch.initial_seed())
print('torch cuda seed: ', torch.cuda.initial_seed())

parser = argparse.ArgumentParser()
args = parser.parse_args("")

args.lr = 0.001
args.epoch_size = 50

args.in_dim = 58
args.hidden_dim1 = 64
args.hidden_dim2 = 256
args.num_layer = 3

args.sc = 'gsc' # 'sc, 'gsc, 'no'
args.optim = 'sgd' #sgd, adam

experiment(args)
