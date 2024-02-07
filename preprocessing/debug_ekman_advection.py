#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Debug Ekman Advection Calculations

Checks differences between calc_ekman_advection (used for the stochastic model paper)
and calc_ekman_advection_htr (which will be used for the re-emergence work)


Created on Wed Feb  7 13:04:36 2024

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

#%% Examine the gradients

# Set other paths
figpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20240207/"
proc.makedir(figpath)

# Set Paths
pathori   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/"
pathnew   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"

# Load calculated gradients
names_old = ["FULL-PIC_Monthly_gradT_lon360.npz","FULL-PIC_Monthly_gradT2_lon360.npz"]
names_new = ["CESM1_HTR_FULL_Monthly_gradT.nc","CESM1_HTR_FULL_Monthly_gradT2.nc"]

#%% First, lets compare the forward difference

# Load the new files, 2 for centered diff, n for new
dsn2 = xr.open_dataset(pathnew+names_new[-1]) # [Ensemble x Month x lat x lon]

# Load the old files
ld_dT2 = np.load(pathori + names_old[-1],allow_pickle=True)
print(ld_dT2.files)

# Place into DataSet
coords_old = {'month':np.arange(1,13,1),'lat':ld_dT2['lat'],'lon':ld_dT2['lon']} # [Month x Lat x Lon]
nmon,nlatg,nlong=ld_dT2['dTdx'].shape

# Make and merge dataarrays
dadx = xr.DataArray(ld_dT2['dTdx'],coords=coords_old,dims=coords_old,name='dTdx2')
dady = xr.DataArray(ld_dT2['dTdy'],coords=coords_old,dims=coords_old,name='dTdy2')
dso2 = xr.merge([dadx,dady])

#Flip Longitude
dso2 = proc.lon360to180_xr(dso2)

#%% Plot differences for each variable (Ens Avg Historical, new script vs PiControl, old script)

mons3  = proc.get_monstr()
kmonth = 1

for kmonth in range(12):
    vlms   = [-1e-4,1e-4]
    vnames = ['dTdx2','dTdy2']
    vnames_fancy = [r"$\frac{d\overline{T}}{dx}$",r"$\frac{d\overline{T}}{dy}$"]
    fnames = ["new, CESM1-Historical","old, CESM1-PiControl"]
    ds_in    =[dsn2.isel(month=kmonth).mean('ensemble'),dso2.isel(month=kmonth)]
    bboxplot = [-80,0,0,65]
    fig,axs,mdict=viz.init_orthomap(2,2,bboxplot,figsize=(10,11))
    #axs = axs.reshape
    
    for ff in range(2):
            
        for vv in range(2):
            
            ax = axs[ff,vv]
            
            ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color='k')
            #ax.set_title("%s (%s)" % (vnames[vv],fnames[ff]),fontsize=24)
            
            ptv =  ds_in[ff][vnames[vv]]
            pcm = ax.pcolormesh(ptv.lon,ptv.lat,ptv,transform=mdict['noProj'],vmin=vlms[0],vmax=vlms[1],cmap="RdBu_r")
            
            #cb = fig.colorbar(pcm,ax=ax,orientation='horizontal',pad=0.01,fraction=0.025)
            
            if ff == 0:
                ax.set_title(vnames_fancy[vv],fontsize=24)
            if vv == 0:
                txt = viz.add_ylabel(fnames[ff],ax=ax,fontsize=20)
                
    cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',pad=0.02,fraction=0.045)
    cb.set_label("SST Gradient ($\degree$C / meter)",fontsize=14)
        
    plt.suptitle("Mean Temperature Gradient for %s (Centered Difference)" % (mons3[kmonth]),fontsize=25)
    
    savename = "%sTemperature_Gradient_Comparison_OldNew_HTR_PiC_mon%02i.png" % (figpath,kmonth+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')
    

#%%








