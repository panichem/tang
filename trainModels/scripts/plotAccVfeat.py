#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:13:19 2024

@author: matt
"""

# imports
import torch
import pickle
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

#from sklearn.metrics import confusion_matrix

# load in torch datasets 
filePaths = sorted(glob.glob('../output/sessions/*True.pt'))

trainAcc = np.empty(25)
testAcc = np.empty(25)
trials = np.empty(25)
neurons = np.empty(25)

for idx, filePath in enumerate(filePaths):
    file = os.path.basename(filePath)
    date = file[0:6]
    print(file)
    
    # load model data
    data = torch.load(filePath)
    
    trainAcc[idx] = data['train_acc'][-1]
    testAcc[idx]  = data['test_acc'][-1]
    
    
    # load training data data
    # load data
    pklFile = open(f"../../../data/torchFiles/{date}.pkl", 'rb')
    spkData = pickle.load(pklFile)
    pklFile.close()
    
    trials[idx], neurons[idx] = spkData['trainData'].dataset.tensors[0].shape
    
    
f, ax = plt.subplots(2, 3,figsize=(16,10))
ax[0,0].plot(trials,trainAcc,'o',label='train')
ax[0,0].plot(trials,testAcc,'o',label='test')
ax[0,0].set_ylabel('proportion correct')
ax[0,0].set_xlabel('nTrials')
ax[0,0].set_title('effect of trials')
ax[0,0].legend(loc='best')


ax[1,0].plot(trials,testAcc-trainAcc,'ok')
ax[1,0].set_ylabel('generalization gap (test-train)')
ax[1,0].set_xlabel('nTrials')    

ax[0,1].plot(neurons,trainAcc,'o',label='train')
ax[0,1].plot(neurons,testAcc,'o',label='test')
ax[0,1].set_ylabel('proportion correct')
ax[0,1].set_xlabel('nNeurons')
ax[0,1].set_title('effect of neurons')
ax[0,1].legend(loc='lower right')


ax[1,1].plot(neurons,testAcc-trainAcc,'ok')
ax[1,1].set_ylabel('generalization gap (test-train)')
ax[1,1].set_xlabel('nNeurons')   

sc   = ax[0,2].scatter(trials,neurons, s = 200, c=testAcc, cmap=plt.cm.jet)
ax[0,2].set_ylabel('nNeurons')
ax[0,2].set_xlabel('nTrials')  
ax[0,2].set_title('trials + neurons')
plt.colorbar(sc, orientation='horizontal',label='test accuracy')    

sc   = ax[1,2].scatter(trials,neurons, s = 200, c=testAcc-trainAcc, cmap=plt.cm.jet)
ax[1,2].set_ylabel('nNeurons')
ax[1,2].set_xlabel('nTrials')  
plt.colorbar(sc, orientation='horizontal',label='generalization gap (test-train)')     
        
plotFileName = "plotAccVfeat.png"
plt.savefig(f"../pics/{plotFileName}")
plt.close()