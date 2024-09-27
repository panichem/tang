#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:24:25 2024

@author: matt
"""

# imports

import torch
import scipy
import pickle
import math
import numpy as np
import glob
import os
from torch.utils.data import TensorDataset

nTrainPerCue = 1000
nTestValPerCue = 100

# load in mat file (from https://discuss.pytorch.org/t/how-to-read-a-dataset-in-mat-form-in-pytorch/71668/2)
filePaths = sorted(glob.glob('../../../data/matFiles/*.mat'))

def bootstrap(a,nIter):
    boot = np.zeros((nIter,a.shape[1]))
    nTrials = a.shape[0]
    for j in range(nIter):
        tmp = a[np.random.choice(nTrials,nTrials),]
        boot[j,] = np.mean(tmp,0)
    return boot

for filePath in filePaths:
    file = os.path.basename(filePath)
    print(f"processing {file}")
    mat = scipy.io.loadmat(filePath)
    spks = mat['spks'] # use the key for data here
    cueAngIdx = mat['cueAngIdx'] # use the key for target here
    tc = mat['tc'] # use the key for target here
    isCorr = mat['isCorr'] # use the key for target here
    
    # filter and format
    
    isCorr = isCorr==1 
    isCorr = np.squeeze(isCorr)
    
    spks = spks[isCorr]
    cueAngIdx = cueAngIdx[isCorr]
    cueAngIdx = cueAngIdx-1
    cueAngIdx = cueAngIdx.squeeze()
    
    tFlt = (tc.squeeze()>1000) & (tc.squeeze()<1400)
    spks = spks[:,tFlt,:]
    spks = np.sum(spks,1) #ntrials x nCells
    
    spks = scipy.stats.zscore(spks)
    spks = np.nan_to_num(spks)
    
    spks = spks.astype('float32')
    
    for i in range(8):
        exec(f"spks{i} = spks[cueAngIdx=={i}]") #grab trials from one condition
        exec(f"np.random.shuffle(spks{i})") #shuffle trials to break order
        exec(f"nTrials = spks{i}.shape[0]") #grab nTrials
        exec('third = math.floor(nTrials/3)')
        exec(f"spks{i}_train, spks{i}_test, spks{i}_val = np.split(spks{i},[third, 2*third])") #split data into thirds
        
        exec(f"spks{i}_train = bootstrap(spks{i}_train,nTrainPerCue)")
        exec(f"spks{i}_test  = bootstrap(spks{i}_test, nTestValPerCue)")
        exec(f"spks{i}_val   = bootstrap(spks{i}_val,  nTestValPerCue)")
    
    spks_train = np.concatenate((spks0_train,spks1_train,spks2_train, 
                                 spks3_train,spks4_train,spks5_train, 
                                 spks6_train, spks7_train),0)
    cueIdx_train = np.concatenate((0*np.ones(nTrainPerCue),
                                  1*np.ones(nTrainPerCue),
                                  2*np.ones(nTrainPerCue),
                                  3*np.ones(nTrainPerCue),
                                  4*np.ones(nTrainPerCue),
                                  5*np.ones(nTrainPerCue),
                                  6*np.ones(nTrainPerCue),
                                  7*np.ones(nTrainPerCue)))
    spks_test = np.concatenate((spks0_test,spks1_test,spks2_test, 
                                 spks3_test,spks4_test,spks5_test, 
                                 spks6_test, spks7_test),0)
    cueIdx_test = np.concatenate((0*np.ones(nTestValPerCue),
                                  1*np.ones(nTestValPerCue),
                                  2*np.ones(nTestValPerCue),
                                  3*np.ones(nTestValPerCue),
                                  4*np.ones(nTestValPerCue),
                                  5*np.ones(nTestValPerCue),
                                  6*np.ones(nTestValPerCue),
                                  7*np.ones(nTestValPerCue)))
    spks_val = np.concatenate((spks0_val,spks1_val,spks2_val, 
                                 spks3_val,spks4_val,spks5_val, 
                                 spks6_val, spks7_val),0)
    cueIdx_val = cueIdx_test
    
    
    # make torch dataset
    data = torch.from_numpy(spks_train.astype('float32'))
    target = torch.from_numpy(cueIdx_train.astype('uint8')) 
    trainData = TensorDataset(data, target)
    
    data = torch.from_numpy(spks_test.astype('float32'))
    target = torch.from_numpy(cueIdx_test.astype('uint8')) 
    testData = TensorDataset(data, target)
    
    data = torch.from_numpy(spks_val.astype('float32'))
    target = torch.from_numpy(cueIdx_val.astype('uint8')) 
    valData = TensorDataset(data, target)
    
    # Open a file and use dump() 
    outFile = f"../../../data/torchFilesBoot/{file[0:-4]}.pkl"
    out = open(outFile, 'wb') 
    pickle.dump({'trainData':trainData,'testData':testData,'valData':valData}, out) 
    out.close()
        
        
        
