#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Average Gradient Question

Essentially:
         __   __       ---------
    is : T  - T_b  =  ( T - T_b) ?
    
    
This came up in the context of the vertical velocity calculations--which gradient
to use?

Created on Tue Apr  1 14:23:08 2025

@author: gliu
"""

import numpy as np
import matplotlib.pyplot as plt

#%% Make Synthetic Timeseries

ntime  = 3600
x      = np.arange(ntime)
scycle = np.sin(2*np.pi / 12 * x)

# Make upper timeseries
T       = np.random.normal(0,1,ntime) + scycle

# Times less realistic scyclem weaker noise
T_b     = -np.roll(T*np.exp(-.08),1) + np.random.normal(0,.5,ntime) + scycle*.8

#%% --- <0> --- Plot Timeseries
fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(8,4.5))
ax.plot(T,label="$T$",marker="o")
ax.plot(T_b,label="$T_b$",marker="d")

ax.grid(True)
ax.legend()

#%%

T_bar = T.mean()
T_b_bar = T_b.mean()

mean_diff = T_b_bar - T_bar


d_dz = T_b - T
ddz_bar = d_dz.mean()


print("Result: ")
print("\tMean of Gradient   : %f" % (ddz_bar))
print("\tMean of Components : %f" % (mean_diff))


"""

Notes, it seems numerically that they are the same and equivalent.
Need to think of a way to write this out mathematically when I have time...

For more info see RN_2025_04

"""



#%% 


