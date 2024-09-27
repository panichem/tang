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
import scipy 

#from sklearn.metrics import confusion_matrix

# load in torch datasets 
subjs = sorted(glob.glob('../../../data/matFiles/*.mat'))

testL  = np.empty(25)
testNL = np.empty(25)
valL = np.empty(25)
valNL = np.empty(25)
bestLin = ['toto']*25
bestNL = ['toto']*25

for iSubj, subjPath in enumerate(subjs):
    subj = os.path.basename(subjPath)
    subj = subj[0:6]
    print(subj)
    
    linFiles = sorted(glob.glob(f"../../trainModels/output/sessions/{subj}*True.pt"))
    testAcc = 0
    valAcc = 0
    for iFile, filePath in enumerate(linFiles):
        # load model data
        data = torch.load(filePath)
        if data['test_acc'][-1] > testAcc:
            testAcc = data['test_acc'][-1]
            valAcc = data['val_acc']
            bestLin[iSubj]= filePath
    testL[iSubj] = testAcc
    valL[iSubj]  = valAcc

    nlFiles = sorted(glob.glob(f"../../trainModels/output/sessions/{subj}*False.pt"))
    testAcc = 0
    valAcc = 0
    for iFile, filePath in enumerate(nlFiles):
        # load model data
        data = torch.load(filePath)
        if data['test_acc'][-1] > testAcc:
            testAcc = data['test_acc'][-1]
            valAcc = data['val_acc']
            bestNL[iSubj] = filePath
    testNL[iSubj] = testAcc
    valNL[iSubj]  = valAcc
        
plt.figure(figsize=(5,5))
plt.scatter(valL,valNL)
plt.xlim([0,1])
plt.ylim([0,1])
plt.plot([0,1],[0,1],'k')
plt.xlabel('epochs')
plt.xlabel('linear (propotion correct)')
plt.ylabel('nonlinear (propotion correct)')
plt.savefig(f"../pics/plotLinNL_sessions")
t = scipy.stats.ttest_rel(valL,valNL)
        

        
        
        
        
        