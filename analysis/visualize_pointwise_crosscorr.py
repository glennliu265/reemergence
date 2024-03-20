#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize
Created on Tue Mar 19 09:31:24 2024

@author: gliu
"""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
from tqdm import tqdm
import copy
import glob

import matplotlib as mpl
mpl.rcParams['font.family'] = 'JetBrains Mono'

#%% Import Custom Modules
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx

#%% Figure Path

datpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
figpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/02_Figures/20240216/"
proc.makedir(figpath)


#%% Import data

ncname = datpath + "CESM1_1920to2005_SSTvSSS_lag00to60_ens01.nc"
ds     = xr.open_dataset(ncname).load()


#%%
im   = 1
dspt = ds.sel(lon=-30,lat=50,method='nearest')

dspt.acf.isel(mons=im,thres=4).plot()

