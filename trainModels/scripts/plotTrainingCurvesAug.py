#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:13:19 2024

@author: matt
"""

# imports
import torch
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

#from sklearn.metrics import confusion_matrix

# load in torch datasets 
filePaths = sorted(glob.glob('../output/sessionsBoot/*.pt'))

trainAcc = np.empty(25)
testAcc = np.empty(25)

for idx, filePath in enumerate(filePaths):
    file = os.path.basename(filePath)
    date = file[0:6]
    print(file)
    
    # load model data
    data = torch.load(filePath)
    
    
    plt.figure()
    plt.plot(data['train_acc'], label='train', c='blue')
    plt.plot(data['test_acc'], label='test', c='red')
    plt.legend(loc='lower right')
    plt.plot(np.array([1, len(data['train_acc'])]),np.ones(2)/8, c = 'black', ls='dashed',label='chance');
    plt.ylabel('proportion correct')
    plt.xlabel('epochs')
    plt.xlim([1, len(data['train_acc'])])

    #plt.title(file[0:6])
    plt.ylim(0,1.01)
     
    plotFileName = f"{file[:-3]}.png"
    plt.savefig(f"../pics/trainingCurvesBoot/{plotFileName}")
    plt.close()