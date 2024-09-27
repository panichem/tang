#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:20:15 2024

@author: matt
"""

import numpy as np
import matplotlib.pyplot as plt


x1 = np.arange(1,10)
y1 = np.ones(len(x1))

x2 = np.arange(3,8)
y2 = 2*np.ones(len(x2))

x3 = np.arange(1.5,9.5)
y3 = 3*np.ones(len(x3))

for x_1 in x1:
    for x_2 in x2:
        plt.plot([x_1, x_2],[1, 2],'-k')
for x_2 in x2:
    for x_3 in x3:
        plt.plot([x_2, x_3],[2, 3],'-k')
    

plt.plot(x1,y1,'o',markersize=20,markerfacecolor='1',markeredgecolor='.3')
plt.plot(x2,y2,'o',markersize=20,markerfacecolor='1',markeredgecolor='.3')
plt.plot(x3,y3,'o',markersize=20,markerfacecolor='1',markeredgecolor='.3')
plt.axis("off")