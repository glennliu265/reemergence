#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize Detrainment Plot
Based on Liu et al. 2023 stochastic model paper (Figure 1).
Copied sections from reemergence/analysis/viz_temp_v_salt.py

Created on Tue Mar 26 08:34:27 2024

@author: gliu
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr
import sys
from tqdm import tqdm
import copy
import glob
import time

# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "Astraeus"

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

# %% Set data paths

# Select Point
lonf   = 330
latf   = 50
locfn, loctitle = proc.make_locstring(lonf, latf)

# Calculation Settings
lags   = np.arange(0,37,1)
lagmax = 3 # Number of lags to fit for exponential function 

# Indicate Paths
datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/lon%s_lat%s/" % (
    lonf, latf)
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20240322/"
proc.makedir(figpath)
outpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/%s/" % locfn

# Other toggles
debug = True # True to make debugging plots

# Plotting Stuff
mons3 = proc.get_monstr(nletters=3)
mpl.rcParams['font.family'] = 'JetBrains Mono'

# Load Mixed Layer Depths
mldpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/mld/"
mldnc   = "CESM1_HTR_FULL_HMXL_NAtl.nc"

# Load and select point
dsh         = xr.open_dataset(mldpath+mldnc)
hbltpt      = dsh.sel(lon=lonf-360, lat=latf,
                 method='nearest').load()  # [Ens x time x z_t]

# Compute Mean Climatology [ens x mon]
hclim       = hbltpt.h.values
lags        = np.arange(61)

# Compute Detrainment month
kprev, _    = scm.find_kprev(hclim.mean(-1)) # Detrainment Months #[12,]
hmax        = hclim.max()#hclim.mean(1).max() # Maximum MLD of seasonal cycle # [1,]

#%% Plotting Parameters

mons3 = proc.get_monstr(nletters=3)
mpl.rcParams['font.family'] = 'Avenir'


nens = hclim.shape[1]


fsz_title = 20
fsz_ticks = 14
fsz_axis  = 16


#%% Make the plot


def monstacker(scycle):
    return np.hstack([scycle,scycle[:1]])
fig,ax     = plt.subplots(1,1,constrained_layout=True,figsize=(12,4.5))

mons3stack = monstacker(mons3)
plotx      = np.arange(1,14)
hstack     = monstacker(hclim.mean(-1))

# ax.fill_between(plotx,0,hstack,alpha=0.50,color="cornflowerblue",zorder=-5)
# ax.fill_between(plotx,hstack,hclim.max()+50,alpha=0.50,color="navy",zorder=-5)

ax = viz.viz_kprev(hclim.mean(1),kprev,ax=ax,lw=3,
                   fsz_lbl=fsz_axis,fsz_axis=fsz_axis,plotarrow=False,msize=15,
                   shade_layers=True)

ax.set_xticklabels(mons3stack,fontsize=fsz_ticks)

ax = viz.add_ticks(ax,minorx=False,grid_col="w",grid_ls='dotted')
ax.set_title("Mixed-Layer Seasonal Cycle and Detrainment Months",fontsize=fsz_title)

# # Plot Detrainment MOnths
# for im in range(12):
#     if kprev[im] == 0:
#         ax.axvline([im+1],ls='solid',c='gray',lw=0.75)

ax.tick_params(axis='both', which='major', labelsize=fsz_ticks)

ax.set_xlabel("Month",fontsize=fsz_axis)
ax.set_ylabel("Mixed-Layer Depth [meters]",fontsize=fsz_axis)
ax.set_ylim([0,175])
ax.invert_yaxis()

savename = "%sDetrainment_plot_fancy_CESM1_HTR_EnsAvg.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)


# plot Correlation


# # for e in range(nens):
# #     plotvar  = monstacker(corr_byens[e,:])
# #     ax.scatter(plotx,plotvar,marker="x")
# # # Plot mean.stdv
# # mu    = monstacker(corr_byens.mean(0))
# # sigma = monstacker(corr_byens.std(0))

# # ax.plot(plotx,mu,lw=3,alpha=1,marker="d",ls='solid',c="k",label="Ens. Mean")
# # ax.fill_between(plotx,mu-sigma,mu+sigma,alpha=0.2,color='gray',label="1$\sigma(Ens.)$")

# ax.set_title("Corr(Detraining SST,Entraining SST)")
# ax.set_ylabel("Correlation")
# ax.set_xlim([1,13])

# for im in range(12):
#     if im == 0:
#         lbl="No Entrainment"
#     else:
#         lbl=""
        
#     if kprev[im] == 0:
#         ax.axvline([im+1],ls='solid',c='gray',lw=0.75,label=lbl)
# ax = viz.add_ticks(ax) 
# ax.set_xticks(plotx,labels=mons3stack) 
# ax.legend() 
    
