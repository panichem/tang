#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 11:29:19 2024

@author: matt

to do: add in linear/non-linear and layer params and use to run a first pass
"""
def trainModelsAug(epochs,hiddenLayers,isLinear):
    #epochs = 300
    #hiddenLayers = []
    #isLinear = True
    
    # imports
    import torch
    import pickle
    import glob
    import os
    import numpy as np
    from torch import nn
    from torch.utils.data import DataLoader
    from torch.utils.data import TensorDataset
    #from sklearn.metrics import confusion_matrix
    
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
        
        trainData = data['trainData']
        origSpks = data['trainData'].dataset.tensors[0][data['trainData'].indices,]
        spks = origSpks
        origLabels = data['trainData'].dataset.tensors[1][data['trainData'].indices]
        labels = origLabels
        nTrials,nNeur = spks.shape
        for i in range(9):
            tmpSpks = origSpks + np.random.normal(np.zeros((nTrials,1)),1*np.ones((1,nNeur))).astype('float32')
            spks = np.concatenate((spks,tmpSpks))
            labels = np.concatenate((labels,origLabels))
        spks = torch.from_numpy(spks)
        labels = torch.from_numpy(labels) 
        trainData = TensorDataset(spks, labels)
            
        
        train_dataloader = DataLoader(trainData, batch_size=batch_size)
        test_dataloader = DataLoader(data['testData'], batch_size=batch_size)
        nNeurons = trainData.tensors[0].shape[1]
    
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
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=.01)
        
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
        
        
        #save model
        outFile = f"../output/sessionsAug/{file[0:-4]}_hl_{sum(hiddenLayers)}_isLin_{str(isLinear)}.pt"
        torch.save({'train_loss':train_loss,'test_loss':test_loss, \
                    'train_acc':train_acc,'test_acc':test_acc, \
                        'model':model.state_dict(), \
                            'optimizer':optimizer.state_dict(), \
                                'test_cm':cm},outFile)
        
        print("Done!")
    
