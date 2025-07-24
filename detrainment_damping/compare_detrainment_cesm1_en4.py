#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compare Detrainment Damping, as estimated from EN4 and CESM1

works with output from:
    - calc_subsurface_damping_en4 (EN4 values)
    - calc_detrainment_correlation_pointwise (CESM1 values)

Created on Tue Jun 24 14:47:55 2025

@author: gliu

"""


import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import time
import sys
import cartopy.crs as ccrs
import glob
import matplotlib as mpl
import tqdm
import pandas as pd

import scipy as sp

#%% Import modules
stormtrack = 0
if stormtrack:
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
    
    # Path to the processed dataset (qnet and ts fields, full, time x lat x lon)
    #datpath =  "/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/hfdamping_RCP85/01_PREPROC/"
    datpath =  "/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/output/anom/"
    figpath =  "/home/glliu/02_Figures/01_WeeklyMeetings/20240621/"
    
else:
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")

    # Path to the processed dataset (qnet and ts fields, full, time x lat x lon)
    datpath =  "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/reanalysis/proc/"
    figpath =  "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/02_Figures/20220511/"
from amv import proc,viz
import scm

import amv.proc as hf # Update hf with actual hfutils script, most relevant functions
import amv.loaders as dl
#%% User Edits

# Plot Settings
mpl.rcParams['font.family'] = 'Avenir'
proj    = ccrs.PlateCarree()
bbplot  = [-80, 0, 35, 75]
mons3   = proc.get_monstr()

figpath = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/02_Figures/20250625/"
proc.makedir(figpath)

#%% Load the NetCDF 

pathen4  = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
ncen4    = "EN4_MIMOC_corr_d_TEMP_detrendbilinear_lagmax3_interp1_ceil0_imshift1_dtdepth1_1979to2021.nc"
ncen4_all = "EN4_MIMOC_corr_d_TEMP_detrendbilinear_lagmax3_interp1_ceil0_imshift1_dtdepth1_1900to2021.nc"
pathcesm = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/ocn_var_3d/"
nccesm   = "CESM1_HTR_FULL_corr_d_TEMP_detrendensmean_lagmax3_interp1_ceil0_imshift1_dtdepth1_ensALL_regridNN.nc"

pathoras =  pathen4
ncoras   = "ORAS5_MIMOC_corr_d_TEMP_detrendRAW_lagmax3_interp1_ceil0_imshift1_dtdepth1_1979to2018_regridERA5.nc"



dsen4    = xr.open_dataset(pathen4 + ncen4).load().lbd_d
dsen4_all = xr.open_dataset(pathen4 + ncen4_all).load().lbd_d
dscesm   = xr.open_dataset(pathcesm + nccesm).load().lbd_d
dsoras   = xr.open_dataset(pathoras + ncoras).load().lbd_d

dsoras = xr.where(dsoras==0.,np.nan,dsoras)

# ----------------------------------------
compare_name = "CESM1_v_EN4_Periods"
inlbds   = [dscesm,dsen4,dsen4_all,dsoras]
expnames_long = ["CESM1 (Ens. Avg.)","EN4 (1979-2021)","EN4 (1900-2021)"]
expnames =["CESM1","EN47921","EN40021"]
expcols       = ["dimgray","orange","red"]

# ----------------------------------------
compare_name = "CESM1_v_EN4_v_ORAS5"
inlbds   = [dscesm,dsen4,dsoras]
expnames_long = ["CESM1 (Ens. Avg.)","EN4 (1979-2021)","ORAS5 (1979-2018)"]
expnames =["CESM1","EN4","ORAS5"]
expcols       = ["dimgray","orange","blue"]

#%% Estimate 
bboxSPGNE       = [-40,-15,52,62]
fsz_tick        = 12
fsz_axis        = 16
imon            = 9

cints = np.arange(-1,1.1,0.1)

spgne_focus = True

if spgne_focus:
    #bboxplot        = [-40,-10,50,65]
    bboxplot        = [-40,-15,52,62]
else:

    bboxplot        = [-50,0,50,65]

for imon in range(12):
    if spgne_focus:
        #fig,axs,mdict   = viz.init_orthomap(1,3,centlon=-25,centlat=55,bboxplot=bboxplot,figsize=(16,8))
       # fig,axs,mdict   = viz.init_orthomap(1,3,centlon=-30,centlat=57,bboxplot=bboxplot,figsize=(16,8))
       fig,axs= plt.subplots(1,3,figsize=(16,8),constrained_layout=True,subplot_kw={'projection':proj})
    else:
        fig,axs,mdict   = viz.init_orthomap(1,3,centlon=-25,centlat=55,bboxplot=bboxplot,figsize=(16,8))
    
    
    for ax in axs:
        
        ax = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=fsz_tick,
                                fix_lon=np.arange(-50,2,2),fix_lat=np.arange(50,67,2),grid_color="k")
        
        viz.plot_box(bboxSPGNE,ax=ax,color='purple',proj=proj,linewidth=2.5)
        
        
    for ii in range(3):
        ax      = axs[ii]
        plotvar = inlbds[ii].isel(mon=imon)
        if ii == 0:
            plotvar = plotvar.mean('ens')
        pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,vmin=0,vmax=1,cmap='inferno')
        
        cl  = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,
                         levels=cints,colors="k",linewidths=0.75)
        clbl = ax.clabel(cl,fontsize=fsz_tick)
        ax.set_title(expnames_long[ii])
        viz.add_fontborder(clbl,w=2.5)
    cb = viz.hcbar(pcm,ax=axs.flatten())
    cb.set_label("%s Subsurface Memory ($\lambda^d$, Correlation)" % mons3[imon],fontsize=fsz_axis)
    
    figname = "%sSubsurface_Damping_Estimates_%s_mon%02i.png" % (figpath,compare_name,imon+1)
    plt.savefig(figname,dpi=150,bbox_inches='tight')


#%% Look at monthly progression, averaged over SPG box

bbsel  = bboxSPGNE
fig,ax = viz.init_monplot(1,1,figsize=(6,4.5))

for ii in range(3):
    plotvar = proc.sel_region_xr(inlbds[ii],bbsel)
    if ii == 0:
        plotvar = plotvar.mean('ens')
    
    _,nlat,nlon = plotvar.shape
    plotvar = plotvar.data.reshape(12,nlat*nlon)
    mu = np.nanmean(plotvar,1)
    sigma = np.nanstd(plotvar,1)
    ax.plot(mons3,mu,label=expnames_long[ii],c=expcols[ii],marker='o')
    ax.fill_between(mons3,mu-sigma,mu+sigma,alpha=0.1,color=expcols[ii])
ax.legend()
ax.set_ylabel("$\lambda^d$ [Correlation]")
ax.set_xlabel("Month of Entrainment")
ax.set_title("Subsurface Damping Averaged over SPGNE")
        
#%%




    