#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Check Daily Mixed-Layer Depth Variability

Created on Thu Sep 12 16:17:53 2024

@author: gliu

"""

import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt

# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "stormtrack"

# First Load the Parameter File
sys.path.append("../")
import reemergence_params as rparams

# Paths and Load Modules
pathdict   = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])
from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx

# Set needed paths
figpath     = pathdict['figpath']
proc.makedir(figpath)
input_path  = pathdict['input_path']
output_path = pathdict['output_path']
procpath    = pathdict['procpath']


#%% Indicate Paths

#figpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20240411/"
datpath   = pathdict['raw_path']
outpath   = pathdict['input_path']+"forcing/"
rawpath   = pathdict['raw_path']


#%%
dpathdaily      = "/stormtrack/data4/glliu/01_Data/CESM1_LE/HMXL/daily/"
ncdaily         = "b.e11.B20TRC5CNBDRD.f09_g16.002.pop.h.nday1.HMXL_2.19200102-20051231.nc"

ds_daily = xr.open_dataset(dpathdaily + ncdaily)


#%% Select a point

lonf = 330
latf = 50
st   = time.time()
hdaily_pt = proc.find_tlatlon(ds_daily,lonf,latf)
hdaily_pt = hdaily_pt.HMXL_2.load()
print("Loaded daily file in %.2fs" % (time.time()-st))

# Need to crop time again (because I think I croppped it incorrectly)
hdaily_crop = hdaily_pt.sel(time=slice("1921-01-01","2005-12-31"))

#%% Group by month
# Compute max, min (range)
# Compute standard deviation (intramonthly)

hd    = hdaily_pt.data # Year x mon x day
ntime = hd.shape[0]


nyr   = int(ntime/(12*30))

hd_reshape = hd.reshape(nyr,12,30)

#%% Do a quick guess

# Compute standard deviation over all days of all years in a given month
hd_monstd = hdaily_pt.groupby('time.month').std('time')

hd_monmax = hdaily_pt.groupby('time.month').max('time')
hd_monmin = hdaily_pt.groupby('time.month').min('time')
hd_monmean = hdaily_pt.groupby('time.month').mean('time')

mons3 = proc.get_monstr()


#%%

fig,ax = viz.init_monplot(1,1)

ax.plot(mons3,hd_monmean,label="Mean",color="k",lw=1.5)
ax.fill_between(mons3,hd_monmean-hd_monstd,hd_monmean+hd_monstd,alpha=0.15)


ax.scatter(mons3,hd_monmax,c='r',marker="x",label="Max",lw=1.5)
ax.scatter(mons3,hd_monmin,c='b',marker="o",label="Min",lw=1.5)
ax.legend()
plt.show()



