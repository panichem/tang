#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:24:25 2024

@author: matt
"""

# imports

import scipy
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import math

    

N = 100; #nneurons
RFs  = np.linspace(0,90-90/N, num = N)
cues = np.arange(0,360,45) #45
kappa = 1;

cues = np.expand_dims(cues,1)
RFs = np.expand_dims(RFs, 0)

meanState = scipy.stats.vonmises.pdf(cues*math.pi/180,kappa,loc=(RFs*math.pi/180))
#meanState = np.cos(cues*math.pi/180-RFs*math.pi/180)

meanState = scipy.stats.zscore(meanState)
meanState = np.nan_to_num(meanState)
#meanState = meanState - np.mean(meanState,axis=0)

pca = PCA(n_components=min(meanState.shape))
pca.fit(meanState)
dat = pca.transform(meanState)

f, ax = plt.subplots(1, 5,figsize=(20,4))

pcNo = np.arange(pca.n_components_) + 1
ax[0].plot(pcNo,np.cumsum(pca.explained_variance_ratio_),'k',marker='.')
ax[0].set_xlabel("principal component")
ax[0].set_ylabel("cumulative variance explained")
ax[0].set_ylim((0,1.1))

hex = ['#1964B0', '#B6DBFF',
       '#00C992', '#386350',
       '#E9DC6D', '#F4A637',
       '#894B45', '#AE75A2']
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
    #ax[idx+1].set_xticks([-10,0,10])
    #ax[idx+1].set_yticks([-10,0,10])
    
plt.tight_layout()
        
#plotFileName = f"{file[:-4]}.png"
#plt.savefig(f"../pics/PCAMeanState/{plotFileName}")

#plt.close()

    
    
    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #ax.scatter(meanState[:,0].T,meanState[:,1].T,meanState[:,2].T)
    
    

