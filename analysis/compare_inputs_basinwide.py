#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compare various inputs of the stochastic model


Created on Thu Feb 15 23:12:17 2024

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

#%% Import Custom Modules
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

sys.path.append(scmpath + '../')

from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx

import stochmod_params as sparams

#%%
figpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/02_Figures/20240216/"
proc.makedir(figpath)

ipath      = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/"
dpath      = ipath + "damping/"
fpath      = ipath + "forcing/"
mpath      = ipath + "mld/"


#%% Helper Functions

def xr_savg(ds):
    # Given xr with [mon] dimension, take seasonal averages
    yr1 = proc.get_xryear()
    
    ds = ds.assign_coords({'mon':yr1}).rename({'mon':'time'})
    ds = ds.groupby('time.season').mean('time')
    return ds


#%% (1) Compare the detrainment damping

ncs   = [
    "CESM1_HTR_FULL_SSS_Expfit_lbdd_maxhclim_lagsfit123_Ens01.nc",
    "CESM1_HTR_FULL_SSS_Expfit_lbdd_monvar_detrendlinear_lagmax3_Ens01.nc",
    "CESM1_HTR_FULL_SST_Expfit_lbdd_maxhclim_lagsfit123_Ens01.nc",
    "CESM1_HTR_FULL_SST_Expfit_lbdd_monvar_detrendlinear_lagmax3_Ens01.nc"
    ]
names  = ["SSS (h max)","SSS (h vary)","SST (h max)","SST (h vary)"]
snames = ["SSSdfix","SSSdvary","SSTdfix","SSTdvary"]
dsall  = [xr.open_dataset(dpath+nc).load() for nc in ncs]
dssavg = [xr_savg(ds) for ds in dsall]

#%% Plotting Parameters

# Plot Parameters
bbplot                      = [-80,0,15,65]
mpl.rcParams['font.family'] = 'Avenir'
sorder                      = ['DJF','MAM','JJA','SON']
fsz_title                   = 20
fsz_axis                    = 16
mons3                       = proc.get_monstr()
#%% Compare the above dampings


timescale     = False
if timescale:
    vunits = "months"
    vlms = [0,48]
else:
    vunits = "mon$^{-1}$"
    vlms = [0,0.3]
    

for im in range(12):
    fig,axs,mdict = viz.init_orthomap(1,4,bbplot,figsize=(16,4))
    proj          = mdict['noProj']
    
    for s in range(4):
        
        invar = dsall[s].lbd_d.isel(mon=im)
        
        lon = invar.lon
        lat = invar.lat
        
        # Set up plot
        ax   = axs[s]
        ax   = viz.add_coast_grid(ax,bbox=bbplot,fill_color="k")
        ax.set_title(names[s],fontsize=fsz_title)
        
        plotvar = invar
        if timescale:
            plotvar = 1/plotvar
        
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,vmin=vlms[0],vmax=vlms[1],zorder=-1)
        #fig.colorbar(pcm,ax=ax,)
    
    cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='vertical',fraction=0.01,pad=0.02)
    cb.set_label("$\lambda^d$ [%s]" % (vunits),fontsize=fsz_axis)
    viz.add_ylabel("%s" % (mons3[im]),ax=axs[0],x=-.15,fontsize=fsz_axis)
    
    
    savename = "%sLbd_d_comparison_timescale%i_mon%02i.png" % (figpath,timescale,im+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')
 

#%% Plot the seasonal Cycle





# Make the plot
vv        = 3
timescale = True
invar     = dssavg[vv].lbd_d

if timescale:
    vunits = "months"
    vlms = [0,48]
else:
    vunits = "mon$^{-1}$"
    vlms = [0,0.3]


fig,axs,mdict = viz.init_orthomap(1,4,bbplot,figsize=(16,4))
proj = mdict['noProj']

for s in range(4):
    
    lon = invar.lon
    lat = invar.lat
    
    
    # Set up plot
    ax   = axs[s]
    ax   = viz.add_coast_grid(ax,bbox=bbplot,fill_color="k")
    seas = sorder[s]
    ax.set_title(seas,fontsize=fsz_title)
    
    plotvar = invar.sel(season=seas)
    if timescale:
        plotvar = 1/plotvar
    
    pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,vmin=vlms[0],vmax=vlms[1],zorder=-1)
    #fig.colorbar(pcm,ax=ax,)

cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='vertical',fraction=0.01,pad=0.02)
cb.set_label("$\lambda^d$ [%s]" % (vunits),fontsize=fsz_axis)
viz.add_ylabel("%s" % (names[vv]),ax=axs[0],x=-.15,fontsize=fsz_axis)


