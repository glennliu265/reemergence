#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Make Quick Re-emergence Plots

Created on Tue Sep  6 21:33:09 2022

@author: gliu
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
import scm

#%%


vnames = ('SST',"SSS")
invars = []
for vname in vnames:
    datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
    ncname  = "HTR-FULL_%s_autocorrelation_thres0.nc" % vname
    ds     = xr.open_dataset(datpath+ncname)
    ldvar    = ds.sel(lon=-30,lat=50,method='nearest')[vname].values
    
    invars.append(ldvar)

lags   = ds.lag.values

#%%

# Autocorrelation Plot parameters
xtk2        = np.arange(0,37,2)
mons3       = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
mons3       = [viz.return_mon_label(m,nletters=3) for m in np.arange(1,13)]
conf        = 0.95
tails       = 2 # Tails for Significance Calculation
alw         = 3 # Autocorrelation Line Width

# Initialize PLot
xtk2       = np.arange(0,37,2)


vcolors=("w","yellow")

ithres      = 2
imonth      = 1




fig,ax = plt.subplots(1,1,figsize=(8,3))




for v in range(2):
    for e in range(42):
        ax.plot(lags,invars[v][ithres,e,:,imonth],alpha=.10,label="",color=vcolors[v])
    
    ax.plot(lags,invars[v][ithres,:,:,imonth].mean(0),label=vnames[v],color=vcolors[v])
    
ax.legend()
    
ax.set_xlim([0,36])
ax.set_xticks(xtk2)
ax.set_ylim([-.1,1])
ax.grid(True,ls='dotted',alpha=0.5)
ax.set_xlabel("Lag (Months)")
ax.set_ylabel("Correlation")