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
from matplotlib.colors import ListedColormap
import math


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
    nTrials, nCells = spks.shape
    
    meanState = np.empty((8,nCells))
    for c in range(8):
        meanState[c,:] = np.mean(spks[cueAngIdx==c,:],0)
    
    
    meanState = scipy.stats.zscore(meanState)
    meanState = np.nan_to_num(meanState)
    
    pca = PCA(n_components=min(meanState.shape))
    pca.fit(meanState)
    dat = pca.transform(meanState)
    
    f, ax = plt.subplots(1, 5,figsize=(20,4))
    
    pcNo = np.arange(pca.n_components_) + 1
    ax[0].plot(pcNo,np.cumsum(pca.explained_variance_ratio_),'k',marker='.')
    ax[0].set_xlabel("principal component")
    ax[0].set_ylabel("cumulative variance explained")
    ax[0].set_ylim((0,1.1))
    
    hex = ['#1964B0', '#B6DBFF','#00C992', '#386350',
                  '#E9DC6D', '#F4A637', '#894B45',
                  '#AE75A2']
    rgb_colors = [tuple(int(h[i:i+2], 16) / 255 for i in (1, 3, 5)) for h in hex]
    #cmap = LinearSegmentedColormap.from_list('mycmap', rgb_colors)
    cmap = ListedColormap(rgb_colors)
    
    #cmap = plt.get_cmap('twilight_shifted',8)
    col = np.arange(8)
    axlim = math.ceil(abs(np.max(dat)))+1
    for idx, startPC in enumerate([0, 2, 4, 6]):
        ax[idx+1].scatter(dat[:,startPC],dat[:,startPC+1],c=col,cmap = cmap)
        ax[idx+1].set_xlabel(f"PC {startPC+1}")
        ax[idx+1].set_ylabel(f"PC {startPC+2}")
        ax[idx+1].set_xlim((-axlim, axlim))
        ax[idx+1].set_ylim((-axlim, axlim))
        ax[idx+1].set_xticks([-10,0,10])
        ax[idx+1].set_yticks([-10,0,10])
        
    plt.tight_layout()
            
    plotFileName = f"{file[:-4]}.png"
    plt.savefig(f"../pics/PCAMeanState/{plotFileName}")
    
    #plt.close()
    
    
    
    
    
    
    
    

