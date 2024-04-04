#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Scrap Script to compare outputs



Created on Mon Apr  1 12:29:04 2024

@author: gliu

"""



import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import os
from tqdm import tqdm

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


rawpath     = pathdict['raw_path']


#%% User Edits

# Indicate information on each of the net Cs
datpathpt   = rawpath + "../../../ptdata/lon330_lat50/"
datpath     = rawpath + "ocn_var_3d/"

# # Temp Comparison
# nc1         = datpath   + "CESM1_HTR_FULL_corr_d_TEMP_detrendensmean_lagmax3_interp1_ceil0_imshift1_dtdepth1_ensALL_regridNN.nc"
# nc2         = datpathpt + "Lbdd_estimate_surface0_imshift1_interpcorr1_TEMP_dtdepth.nc"


# # SALT comparison (ens kprev variation)
# comparename = "SALT_kprevbyens"
# nc1         = datpath   + "CESM1_HTR_FULL_corr_d_SALT_detrendensmean_lagmax3_interp1_ceil0_imshift1_dtdepth1_ensALL_regridNN.nc"
# nc2         = datpathpt + "Lbdd_estimate_surface0_imshift1_interpcorr1_kprevbyens0_SALT_dtdepth.nc"
# nc3         = datpathpt + "Lbdd_estimate_surface0_imshift1_interpcorr1_kprevbyens1_SALT_dtdepth.nc"

# # TEMP comparison (ens kprevvariation)
# comparename = "TEMP_kprevbyens"
# nc1         = datpath   + "CESM1_HTR_FULL_corr_d_TEMP_detrendensmean_lagmax3_interp1_ceil0_imshift1_dtdepth1_ensALL_regridNN.nc"
# nc2         = datpathpt + "Lbdd_estimate_surface0_imshift1_interpcorr1_kprevbyens0_TEMP_dtdepth.nc"
# nc3         = datpathpt + "Lbdd_estimate_surface0_imshift1_interpcorr1_kprevbyens1_TEMP_dtdepth.nc"

# # Salt Comparison
# ncs         = [nc1,nc2,nc3]
# expnames    = ["Pointwise Script","Individual Script (Ens Avg. MLD)","Individual Script (by Member)"] 
# expcols     = ["midnightblue","hotpink","limegreen"]
# expmarkers  = ["d","x","+"]

# TEMP comparison (dtdepth)
comparename = "TEMP_dtdepth"
nc1         = datpath   + "CESM1_HTR_FULL_corr_d_TEMP_detrendensmean_lagmax3_interp1_ceil0_imshift1_dtdepth1_ensALL_regridNN.nc"
nc3         = datpathpt + "Lbdd_estimate_surface0_imshift1_interpcorr1_kprevbyens1_TEMP.nc"
expnames    = ["Detrainment Depth","Depth of Surrounding Months"] 
expcols     = ["midnightblue","hotpink",]
expmarkers  = ["d","x",]
ncs         = [nc1,nc3]




lonf        = -30
latf        = 50


# Plotting Params
# Labels
mons3 = proc.get_monstr(nletters=3)
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['font.family']      = 'JetBrains Mono'#'Courier'#'STIXGeneral'



#%% Load the Files

nexps   = len(ncs)

dsall   = []
for n in range(nexps):
    
    
    if n == 0: # Get point data from pointwise script
        ds = xr.open_dataset(ncs[n]).lbd_d.load()
        ds = proc.selpt_ds(ds,lonf,latf)
    else:
        ds = xr.open_dataset(ncs[n]).corr_d.load()
    dsall.append(ds)

#%% Make the comparison

def plotens(x,ds,ax,c='k',alpha=0.1,label=None):
    
    nens = len(ds.ens)
    for e in range(nens):
        ax.plot(x,ds.isel(ens=e),c=c,alpha=alpha,label="")
    
    mu      = ds.mean('ens')
    sigma   = ds.std('ens')
    ax.plot(x,mu,label=label,c=c,lw=2.5)
    #ax.fill_between(x,mu-sigma,mu+sigma,alpha=0.1,color=c)
    return mu,sigma,ax

fig,ax = viz.init_monplot(1,1,figsize=(12,4.5))
for exp in range(nexps):
    # if exp !=0:
    #     continue
    
    ds      = dsall[exp]
    x       =  mons3
    label   = expnames[exp]
    c       = expcols[exp]
    ds      = xr.where(ds==0.,np.nan,ds)
    mu      = ds.mean('ens')
    sigma   = ds.std('ens')
    print("Sigma for %s is %s" % (label,sigma))
    
    for e in range(42):
        ax.scatter(x,ds.isel(ens=e),c=c,alpha=0.1,label="",zorder=-2,marker=expmarkers[exp])
        
    
    print(exp)
    print(mu)
    ax.plot(x,mu,label=label,c=c,lw=2.5)
    
    ax.legend()
    
    
    ax.fill_between(x,mu-sigma,mu+sigma,alpha=0.1,color=c)
    # mu,sigma,ax = plotens(mons3,dsall[exp],ax,c=expcols[exp],label=expnames[exp])

ax.set_ylabel("Corr(Detrain,Entrain-1)")
ax.set_ylim([0,1])
ax.set_xlabel("Month of Entrainment")
savename = "%sPointwise_vs_Pt_Script_Comparison_%s.png" % (figpath,comparename)
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)

#%%

e = 9 # it is only different for ensemx`ble member 9?
#e = 21

fig,ax = viz.init_monplot(1,1,figsize=(12,4.5))
for exp in range(nexps):
    
    ds      = dsall[exp]
    x       =  mons3
    label   = expnames[exp]
    c       = expcols[exp]
    mu      = ds.isel(ens=e)
    if exp > 1:
        ls = "dashed"
    else:
        ls="solid"
    print(mu)
    #sigma   = ds.std('ens')
    #print("Sigma for %s is %s" % (label,sigma))
    # for e in range(42):
    #     ax.scatter(x,ds.isel(ens=e),c=c,alpha=0.1,label="",zorder=-2,marker=expmarkers[exp])
        
    ax.plot(x,mu,label=label,c=c,lw=2.5,ls=ls)
    
ax.legend()
    #ax.fill_between(x,mu-sigma,mu+sigma,alpha=0.1,color=c)
    #mu,sigma,ax = plotens(mons3,dsall[exp],ax,c=expcols[exp],label=expnames[exp])
    
    

#%% 


