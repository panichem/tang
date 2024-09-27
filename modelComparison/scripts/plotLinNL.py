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
subjs = sorted(glob.glob('../../../data/matFiles/*.mat'))

testL  = np.empty(25)
testNL = np.empty(25)
valL = np.empty(25)
valNL = np.empty(25)

for iSubj, subjPath in enumerate(subjs):
    subj = os.path.basename(subjPath)
    subj = subj[0:6]
    print(subj)
    
    linFiles = sorted(glob.glob(f"../../trainModels/output/sessionsBoot/{subj}*True.pt"))
    testAcc = 0
    valAcc = 0
    for iFile, filePath in enumerate(linFiles):
        # load model data
        data = torch.load(filePath)
        if data['test_acc'][-1] > testAcc:
            testAcc = data['test_acc'][-1]
            valAcc = data['val_acc']
    testL[iSubj] = testAcc
    valL[iSubj]  = valAcc

    nlFiles = sorted(glob.glob(f"../../trainModels/output/sessionsBoot/{subj}*False.pt"))
    testAcc = 0
    valAcc = 0
    for iFile, filePath in enumerate(nlFiles):
        # load model data
        data = torch.load(filePath)
        if data['test_acc'][-1] > testAcc:
            testAcc = data['test_acc'][-1]
            valAcc = data['val_acc']
    testNL[iSubj] = testAcc
    valNL[iSubj]  = valAcc
        
        
        
        
        # plt.figure()
        # plt.plot(data['train_acc'], label='train', c='blue')
        # plt.plot(data['test_acc'], label='test', c='red')
        # plt.legend(loc='lower right')
        # plt.plot(np.array([1, len(data['train_acc'])]),np.ones(2)/8, c = 'black', ls='dashed',label='chance');
        # plt.ylabel('proportion correct')
        # plt.xlabel('epochs')
        # plt.xlim([1, len(data['train_acc'])])
    
        # #plt.title(file[0:6])
        # plt.ylim(0,1.01)
         
        #plotFileName = f"{file[:-3]}.png"
        #plt.savefig(f"../pics/trainingCurvesBoot/{plotFileName}")
        #plt.close()