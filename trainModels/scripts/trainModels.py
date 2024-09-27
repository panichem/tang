#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 11:29:19 2024

@author: matt

to do: add in linear/non-linear and layer params and use to run a first pass
"""
# imports
import torch
import pickle
import glob
import os
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import argparse
#from sklearn.metrics import confusion_matrix

debug = False

if debug:
    hiddenLayers = [0]
    isLinear = True
    learnRate = -1
    l2 = -1
    epochs = 50
    
    # load in torch datasets 
    saveData = False
else:
    #read in arguments
    def list_of_ints(arg):
        return list(map(int,arg.split(',')))
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--hiddenLayers', type=list_of_ints)
    parser.add_argument('--isLin', dest='isLin', action='store_true',
                        help='Set the flag value to True.')
    parser.add_argument('--no-isLin', dest='isLin', action='store_false',
                        help='Set the flag value to False.')
    parser.add_argument('--learnRate',type=int) #-2
    parser.add_argument('--l2',type=int) #-2
    
    args = parser.parse_args()
    epochs = args.epochs
    if args.hiddenLayers is None:
        args.hiddenLayers = []    
    hiddenLayers = args.hiddenLayers
    isLinear = args.isLin
    learnRate = args.learnRate
    l2 = args.l2

    print(args.isLin)
    print(type(args.isLin))

    # load in torch datasets 
    saveData = True

# load in torch datasets 
filePaths = sorted(glob.glob('../../../data/torchFiles/*.pkl'))

for filePath in filePaths:
    file = os.path.basename(filePath)
    print(file)
    
    # load data
    pklFile = open(filePath, 'rb')
    data = pickle.load(pklFile)
    pklFile.close()
    
    # Create data loaders.
    batch_size = 64
    train_dataloader = DataLoader(data['trainData'], batch_size=batch_size)
    test_dataloader  = DataLoader(data['testData'], batch_size=batch_size)
    val_dataloader   = DataLoader(data['valData'], batch_size=batch_size)
    nNeurons = data['trainData'].dataset.tensors[0].shape[1]

    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    
    # Define model - need to update hardcoding of layers/linearity later
    class NeuralNetwork(nn.Module):
        
        def __init__(self, nInput, nHidden, isLin):
            super().__init__()
            self.flatten = nn.Flatten()
            self.mlp = nn.Sequential() # Initialize layers of MLP
            
            inNum = nInput
            if nHidden[0] != 0:
                for i in range(len(nHidden)):
                    outNum = nHidden[i]
                    layer = nn.Linear(inNum,outNum)
                    inNum = outNum
                    self.mlp.add_module('linear_%d'%i, layer) 
                    if ~isLin:
                        self.mlp.add_module('relu_%d'%i, nn.ReLU()) 
                
            outLayer = nn.Linear(inNum, 8) # Create final layer
            self.mlp.add_module('linear_out', outLayer) # Append the final layer
        
    
        def forward(self, x):
            x = self.flatten(x)
            logits = self.mlp(x)
            return logits   
        
    model = NeuralNetwork(nNeurons,hiddenLayers,isLinear).to(device)
    #print(model)
    
    #set training params
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=10**learnRate, weight_decay=10**l2)

    
    #define training function
    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        train_loss, correct = 0, 0
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
    
            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)
            train_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
        train_loss /= len(dataloader)
        correct /= size
        return train_loss, correct
    
    #define test function 
    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct, cm = 0, 0, np.zeros([8,8])
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                #cm += confusion_matrix(y.cpu().numpy(), pred.argmax(1).cpu().numpy())
        test_loss /= num_batches
        correct /= size
        return test_loss, correct, cm
    
    #train and test
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    
    for t in range(epochs):
        tr_loss, tr_correct = train(train_dataloader, model, loss_fn, optimizer)
        ts_loss, ts_correct, cm = test(test_dataloader, model, loss_fn)
        
        train_loss.append(tr_loss)
        test_loss.append(ts_loss)
        train_acc.append(tr_correct)
        test_acc.append(ts_correct)
    val_loss, val_acc, val_cm = test(val_dataloader, model, loss_fn)
    
    
    # save model
    outFile = f"../output/sessions/{file[0:-4]}_hl_{'_'.join(str(i) for i in hiddenLayers)}_epochs_{str(epochs)}_lr_{str(learnRate)}_l2_{str(l2)}_isLin_{str(isLinear)}.pt"
    if saveData:
        torch.save({'train_loss':train_loss,'test_loss':test_loss, \
                    'train_acc':train_acc,'test_acc':test_acc, \
                        'val_loss':val_loss,'val_acc':val_acc, \
                           'model':model.state_dict(), \
                               'optimizer':optimizer.state_dict(), \
                                   'test_cm':cm},outFile)
    
    print("Done!")

