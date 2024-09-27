#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:24:25 2024

@author: matt
"""

# imports

import scipy
import numpy as np
import glob
import os
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# load in mat file (from https://discuss.pytorch.org/t/how-to-read-a-dataset-in-mat-form-in-pytorch/71668/2)
filePaths = sorted(glob.glob('../../../data/matFiles/210921.mat'))

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
    
    print(spks.shape)
    pca = PCA(n_components=min(spks.shape))
    pca.fit(spks)
    
    #x = np.arange(pca.n_components_) + 1
    #plt.plot(x,np.cumsum(pca.explained_variance_ratio_))
    tmpIdx = np.argmax(np.cumsum(pca.explained_variance_ratio_)>.8)
    spks2 = spks[:,:tmpIdx]
    #dat = pca.transform(spks)
