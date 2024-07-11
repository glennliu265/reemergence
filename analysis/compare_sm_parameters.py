#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compare Stochastic Model Parameters

Currently Works with CESM1 vs CESM2 PIC)

Created on Fri Jul  5 13:29:18 2024

@author: gliu

"""

import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
from scipy.io import loadmat

import matplotlib.pyplot as plt
import sys
import glob
import os

import tqdm
import time

# ----------------------------------
# %% Import custom modules and paths
# ----------------------------------

# Import re-eergemce parameters

# Indicate the Machine!
machine = "Astraeus"

# First Load the Parameter File
cwd = os.getcwd()
sys.path.append(cwd+ "/..")
import reemergence_params as rparams

# Paths and Load Modules
pathdict = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])

# Set needed paths
figpath     = pathdict['figpath']
input_path  = pathdict['input_path']
output_path = pathdict['output_path']
procpath    = pathdict['procpath']
rawpath     = pathdict['raw_path']


#%% Import Custom Modules

# Import AMV Calculation
from amv import proc,viz
import amv.loaders as dl

# Import stochastic model scripts
import scm


#%% Indicate Plotting Parameters (taken from visualize_rem_cmip6)


bboxplot        = [-80,0,10,65]
mpl.rcParams['font.family'] = 'Avenir'
mons3           = proc.get_monstr(nletters=3)

fsz_tick        = 18
fsz_axis        = 20
fsz_title       = 28

rhocrit = proc.ttest_rho(0.05,2,86)

proj    = ccrs.PlateCarree()


#%% User Edits

# -----------------------------------------------------
#%% CESM1 (copied from run_sm_rewrite, sm_rewrite_loop)
# -----------------------------------------------------

# I think these should be the correct settings for the default run...
method      = 5 # 1 = No Testing; 2 = SST autocorr; 3 = SST-FLX crosscorr, 4 = Both, 5 - replace insignificant values in FULL with those from SLAB
lagstr      = "lag1"
ensorem     = True
mconfig     = "SLAB_PIC"
frcname     = 'flxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0'
projpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
datpath     = projpath + '01_Data/'
input_path  = datpath + 'model_input/'

lon,lat,h,kprevall,damping,dampingfull,alpha,alpha_full = scm.load_inputs(mconfig,frcname,input_path,
                                                                      load_both=True,method=method,
                                                                      lagstr=lagstr,ensorem=ensorem)

# Put into DataArray
mldcoords       = dict(lon=lon,lat=lat,mon=np.arange(1,13,1))
fpcoords        = dict(lon=lon,lat=lat,mode=np.arange(1,alpha_full.shape[2]+1),mon=np.arange(1,13,1))
mld_cesm1       = xr.DataArray(h,coords=mldcoords,dims=mldcoords,name="h")
hff_cesm1       = xr.DataArray(dampingfull,coords=mldcoords,dims=mldcoords,name="damping")
fprime_cesm1    = xr.DataArray(alpha_full,coords=fpcoords,dims=fpcoords,name="Fprime")

# -----------------------------------------------------
#%% Load CESM2
# -----------------------------------------------------

input_path  = pathdict['input_path']

# Mixed Layer Depth ------
nc1         = "cesm2_pic_HMXL_NAtl_0200to2000.nc"
path1       = input_path + "mld/"
mld_cesm2   = xr.open_dataset(path1+nc1).load().h

# HFF
nc1         = "cesm2_pic_qnet_damping_CESM2PiCqnetDamp.nc"
path1       = input_path + "damping/"
hff_cesm2   = xr.open_dataset(path1+nc1).load().damping

# 
nc1         = "cesm2_pic_Fprime_EOF_corrected_CESM2PiCqnetDamp_nroll0_perc090_NAtl_EnsAvg.nc"
path1       = input_path + "forcing/"
fprime_cesm2 = xr.open_dataset(path1+nc1).load()

# -----------------------------------------------------
#%% Resize the DS
# -----------------------------------------------------

mlds        = proc.resize_ds([mld_cesm1.transpose('mon','lat','lon'),mld_cesm2])
hffs        = proc.resize_ds([hff_cesm1.transpose('mon','lat','lon'),hff_cesm2])
fprimes     = proc.resize_ds([fprime_cesm1.transpose('mon','mode','lat','lon'),fprime_cesm2.Fprime])


fprimes_mag = [(fp**2).sum('mode')**(1/2) for fp in fprimes]

exnames     = ["CESM1 (400-2200)","CESM2 (200-2000)","CESM2 - CESM1"]


#%% Make a mask

mldmask = [xr.where(~np.isnan(ds.sum('mon',skipna=False)),1,np.nan) for ds in mlds]
mldmask = mldmask[0].data * mldmask[1].data # xr.DataArray(mldmask[0].data * mldmask[1].data,coords=mldmask[0].coords,)

hffmask = [xr.where(~np.isnan(ds.sum('mon',skipna=False)),1,np.nan) for ds in hffs]
hffmask = hffmask[0].data * hffmask[1].data

fpmask  = [xr.where(~np.isnan(ds.sum('mon',skipna=False)),1,np.nan) for ds in fprimes_mag]
fpmask  = fpmask[0].data * fpmask[1].data

maskall  = xr.DataArray(mldmask*hffmask*fpmask,coords=mlds[0].sum('mon').coords)

# ------------------------------------------------------
#%% Make the comparison plots 
# ------------------------------------------------------

# %% MLD
bbox_mld = [-80,0,30,65]
selmons  = [1,]

# Max MLD
fig,axs,_ = viz.init_orthomap(1,3,bbox_mld,figsize=(18.5,5))

for aa in range(3):
    
    ax      = axs[aa]
    ax      = viz.add_coast_grid(ax,bbox_mld,fill_color="lightgray",fontsize=20,
                 fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
    ax.set_title(exnames[aa],fontsize=fsz_title)
    
    if aa < 2:
        plotvar = mlds[aa].isel(mon=selmons).mean('mon')
        cmap    = 'cmo.deep'
        vlms    = [0,500]
        cints_deep = np.arange(500,1600,250)
    else:
        plotvar    = mlds[1].isel(mon=selmons).mean('mon') - mlds[0].isel(mon=selmons).mean('mon')
        cmap       = "cmo.balance"
        vlms       = [-200,200]
        
        cints_deep = np.arange(-600,100,100)
    
    
    pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar * maskall,transform=proj,
                        cmap=cmap,vmin=vlms[0],vmax=vlms[1])
    cb  = viz.hcbar(pcm,ax=ax,fraction=0.045)
    #cb.tick_params()
    
    #if aa < 2:
    cl  = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                     levels=cints_deep,colors="w",linewidths=0.75)
    ax.clabel(cl)

plt.suptitle("%s Mixed-Layer Depths (m), PiControl" % mons3[selmons[0]],fontsize=28)
        
savename = "%sMLD_CESM1_v_CESM2_PiControl_mon%02i.png" % (figpath,selmons[0])
plt.savefig(savename,dpi=150)


#%% HFF

bbox_mld    = [-80,0,30,65]
selmons     = [1,]

fig,axs,_   = viz.init_orthomap(1,3,bbox_mld,figsize=(18.5,5))

for aa in range(3):
    
    ax      = axs[aa]
    ax      = viz.add_coast_grid(ax,bbox_mld,fill_color="lightgray",fontsize=20,
                 fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
    ax.set_title(exnames[aa],fontsize=fsz_title)
    
    if aa < 2:
        plotvar = hffs[aa].isel(mon=selmons).mean('mon')
        cmap    = "cmo.balance"
        vlms    = [-40,40]
        #cints_deep = np.arange(500,1600,250)
    else:
        plotvar    = hffs[1].isel(mon=selmons).mean('mon') - hffs[0].isel(mon=selmons).mean('mon')
        cmap       = "cmo.balance"
        vlms       = [-15,15]
        
        #cints_deep = np.arange(-600,100,100)
    
    
    pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar*maskall,transform=proj,
                        cmap=cmap,vmin=vlms[0],vmax=vlms[1])
    cb  = viz.hcbar(pcm,ax=ax,fraction=0.045)
    #cb.tick_params()
    
    # #if aa < 2:
    # cl  = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
    #                  levels=cints_deep,colors="w",linewidths=0.75)
    # ax.clabel(cl)

plt.suptitle("%s Heat Flux Feedback ($W/m^2/\degree C$), PiControl" % mons3[selmons[0]],fontsize=28)
        
savename = "%sHFF_CESM1_v_CESM2_PiControl_mon%02i.png" % (figpath,selmons[0])
plt.savefig(savename,dpi=150)

#%% Forcing

bbox_mld    = [-80,0,30,65]
selmons     = [1,]

fig,axs,_   = viz.init_orthomap(1,3,bbox_mld,figsize=(18.5,5))

for aa in range(3):
    
    ax      = axs[aa]
    ax      = viz.add_coast_grid(ax,bbox_mld,fill_color="lightgray",fontsize=20,
                 fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
    ax.set_title(exnames[aa],fontsize=fsz_title)
    
    if aa < 2:
        plotvar = fprimes_mag[aa].isel(mon=selmons).mean('mon')
        cmap    = "inferno"
        vlms    = [0,100]
        #cints_deep = np.arange(500,1600,250)
    else:
        plotvar    = fprimes_mag[1].isel(mon=selmons).mean('mon') - fprimes_mag[0].isel(mon=selmons).mean('mon')
        cmap       = "cmo.balance"
        vlms       = [-20,20]
        
        #cints_deep = np.arange(-600,100,100)
    
    
    pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar*maskall,transform=proj,
                        cmap=cmap,vmin=vlms[0],vmax=vlms[1])
    cb  = viz.hcbar(pcm,ax=ax,fraction=0.045)
    #cb.tick_params()
    
    # #if aa < 2:
    # cl  = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
    #                  levels=cints_deep,colors="w",linewidths=0.75)
    # ax.clabel(cl)

plt.suptitle("%s Forcing Amplitude ($W/m^2$), PiControl" % mons3[selmons[0]],fontsize=28)

savename = "%sFprime_CESM1_v_CESM2_PiControl_mon%02i.png" % (figpath,selmons[0])
plt.savefig(savename,dpi=150)

#%% Examine the parameters averaged over the selected box

    