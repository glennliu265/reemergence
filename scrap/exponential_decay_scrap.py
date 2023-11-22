#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Decay for Testing Exponential Decay

Created on Mon Nov 13 13:22:04 2023

@author: gliu
"""

import numpy as np
import xarray as xr
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy import stats,signal
from tqdm import tqdm
import glob

import sys
stormtrack = 1
if stormtrack == 0:
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
elif stormtrack == 1:
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
from amv import proc
import time
import yo_box as ybx
import tbx
import pandas as pd

#%% Ok Whatever BS

def init_splot(ax=None,fig=None,figsize=(6,4)):
    if (ax is None) and (fig is None):
        fig,ax=plt.subplots(1,1,figsize=figsize,constrained_layout=True)
    elif (ax is None):
        ax = plt.gca()
        
    return fig,ax


#%% Generate White Noise
Fp   = np.random.normal(0,1,(120000))
#%% Set Some Parameters

h     = np.array([120,130,120,110,100,90,80,70,80,90,100,110])-20

months = np.arange(1,13)


fig,axs = plt.subplots(3,1,constrained_layout=True,figsize=(8,6))

ax  = axs[0]
ax.set_xticks(np.arange(0,14,1))
ax.plot(months,h,marker="o",label="MLD")
ax.legend()

#%% Integrate a Theoretical Value


#%% Exponential Decay Demo

T0              = 22
decay_timescale = 12 # Months
Tdexp           = 1/decay_timescale
lags            = np.arange(0,25)

acf_theory      = T0 * np.exp(-Tdexp * lags)

fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(8,4))

ax.plot(lags,acf_theory,color="k",label=r"$e^{-t \, \tau^{-1}}$",marker="d")

decay_value = T0 * np.exp(-1)
ax.axhline([decay_value],color='gray',label="%.2f" % decay_value,ls='dashed')
ax.axvline([decay_timescale],color='green',ls='dotted',
           label=r"Decay Timescale ($\tau$=%s)" % decay_timescale)
ax.legend()
ax.grid(True,ls='dotted')




