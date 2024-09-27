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
import numpy as np
import glob
import os
from torch.utils.data import TensorDataset

# load in mat file (from https://discuss.pytorch.org/t/how-to-read-a-dataset-in-mat-form-in-pytorch/71668/2)
filePaths = sorted(glob.glob('../../../data/matFiles/*.mat'))

for iFile, filePath in enumerate(filePaths):
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
    
    if iFile == 0:
        omniSpks = spks
        To, No = omniSpks.shape
        omniCueAngIdx = cueAngIdx
    else:
        T, N = spks.shape
        top = np.concatenate(    ( omniSpks,       np.zeros((To,N)) ),1)
        bottom = np.concatenate( ( np.zeros((T,No)), spks           ),1)
        omniSpks = np.concatenate((top,bottom),0)
        To, No = omniSpks.shape
        
        omniCueAngIdx = np.concatenate((omniCueAngIdx,cueAngIdx),0)
            
        
    
# make torch dataset
omniSpks = omniSpks.astype('float32')

data = torch.from_numpy(omniSpks)
target = torch.from_numpy(omniCueAngIdx) 
dataset = TensorDataset(data, target)

# do splits

seed1 = torch.Generator().manual_seed(42)
toto = torch.utils.data.random_split(dataset, [.8, .1, .1], generator=seed1)
trainData = toto[0]
testData = toto[1]
valData = toto[2]
    
# Open a file and use dump() 
outFile = "../../../data//torchFilesOmni/omni.pkl"
out = open(outFile, 'wb') 
pickle.dump({'trainData':trainData,'testData':testData,'valData':valData}, out) 
out.close()
        
        
        
