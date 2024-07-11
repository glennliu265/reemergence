#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compare Pointwise SST-SSS Cross Correlation for CESM1 and the Stochastic Model

- Works with ouput from [pointwise_crosscorrelation.py]
- Also works with high-pass output from [high_pass_correlations.py]

Created on Fri Jun 14 16:06:51 2024

@author: gliu

"""

import sys
import copy
import glob
import time

from tqdm import tqdm

import numpy as np
import xarray as xr
import matplotlib as mpl
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import os


# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "Astraeus"

# First Load the Parameter File
cwd = os.getcwd()
sys.path.append(cwd+"/../")
import reemergence_params as rparams

# Paths and Load Modules
pathdict   = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])
sys.path.append(pathdict['scmpath'] + "../")
from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx
import stochmod_params as sparams

# Set needed paths
figpath     = pathdict['figpath']
input_path  = pathdict['input_path']
output_path = pathdict['output_path']
procpath    = pathdict['procpath']
rawpath     = pathdict['raw_path']

# Make Needed Paths
proc.makedir(figpath)


#%% Load the files 

ncs      = ["CESM1_1920to2005_SST_SSS_crosscorrelation_nomasklag1_nroll0_lag00to60_ALL_ensALL.nc",
            "SM_SST_SSS_lbdE_neg_crosscorrelation_nomasklag1_nroll0_lag00to60_ALL_ensALL.nc"]

expnames = ["CESM1","Stochastic Model"]


nexps       = len(ncs)
ccfs        = []
for ex in range(nexps):
    ds = xr.open_dataset(procpath+ncs[ex]).acf.load()
    ccfs.append(ds)

#%% Plotting options

bboxplot                    = [-80,0,20,65]
mpl.rcParams['font.family'] = 'Avenir'
mons3                       = proc.get_monstr(nletters=3)
fsz_tick                    = 18
fsz_axis                    = 14
fsz_title                   = 16

proj                        = ccrs.PlateCarree()

#%% Visualize things at a few lags
for kmonth in range(12):
    sellags   = [0,6,12]
    
    cints     = np.arange(-1,1.1,0.1)
    
    fig,axs,_ = viz.init_orthomap(2,3,bboxplot,figsize=(18,8),)
    
    for ex in range(2):
        for ll in range(3):
            
            lag     = sellags[ll]
            ax      = axs[ex,ll]
            
            plotvar = ccfs[ex].isel(lags=lag,mons=kmonth).squeeze().mean('ens')
            # pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar.values.T,transform=proj,
            #                         vmin=-1,vmax=1,cmap='cmo.balance')
            pcm = ax.contourf(plotvar.lon,plotvar.lat,plotvar.values.T,transform=proj,levels=cints,cmap='cmo.balance')
            cl  = ax.contour(plotvar.lon,plotvar.lat,plotvar.values.T,transform=proj,levels=cints,colors='dimgray',
                             linewidths=0.55,)
            
            ax.clabel(cl,cints[::2])
            ax      = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
            
            #cb = viz.hcbar(pcm,ax=ax)
            
            if ex == 0:
                ax.set_title("Lag %02i" % (lag))
            if ll == 0:
                viz.add_ylabel(expnames[ex],ax=ax)
    
    cb = viz.hcbar(pcm,ax=axs.flatten(),fraction=0.045,pad=0.01)
    cb.set_label("SST-SSS Cross Correlation (%s)" % (mons3[kmonth]))


    savename = "%sSST_SSS_Cross_Correlation_CESMvSM_mon%02i.png" % (figpath,kmonth+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')
#%% Make the same plot as above, but focus on averages over a season

# Select the months to average over
selmon = [8,9,10]
avgstr = ''.join([mons3[a][0] for a in selmon])

# Select the lags
lag    = 0



fig,axs,_ = viz.init_orthomap(1,2,bboxplot,figsize=(10,5),)
for ex in range(2):
    
    ax      = axs[ex]
    
    plotvar = ccfs[ex].isel(lags=lag,mons=selmon).squeeze().mean('mons').mean('ens')
    
    pcm = ax.contourf(plotvar.lon,plotvar.lat,plotvar.values.T,transform=proj,levels=cints,cmap='cmo.balance')
    cl  = ax.contour(plotvar.lon,plotvar.lat,plotvar.values.T,transform=proj,levels=cints,colors='dimgray',
                     linewidths=0.55,)
    
    ax.clabel(cl,cints[::2])
    ax      = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
    
    ax.set_title(expnames[ex],fontsize=fsz_axis)
    if ll == 0:
        viz.add_ylabel(expnames[ex],ax=ax)
        
plt.suptitle("Cross-Correlation (SSS Lagged %02i Months, %s Avg.)" % (lag,avgstr,),fontsize=fsz_title)
cb = viz.hcbar(pcm,ax=axs.flatten(),fraction=0.045,pad=0.01)
cb.set_label("SST-SSS Cross Correlation (%s)" % (mons3[kmonth]),fontsize=fsz_axis)   
    
savename = "%sSST_SSS_Cross_Correlation_CESMvSM_%s_lag%02i.png" % (figpath,avgstr,lag)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% 

"""

Section II. Compare Cross-Correlation (all, instantaneous) for regular and high-pass filtered results
Works with output from high_pass_correlations

"""

hicutoff  = 12
expnames  = ["CESM1_LE","SM_Default","SM_LHFLX"]
plotnames = ["CESM1","Stochastic Model","Stochastic Model (LHFLX Only)"] 
hppath    = rawpath + "filtered/"

nexps  = len(expnames)
ds_raw = []
ds_hp  = []
for ex in range(nexps):
    
    ncname  = hppath + "%s_SST_SSS_NATL_crosscorr_raw.nc" % (expnames[ex],)
    ds      = xr.open_dataset(ncname).corr.load()
    ds_raw.append(ds.copy())
    
    ncname1 = hppath + "%s_SST_SSS_NATL_crosscorr_hpf%02imon.nc" % (expnames[ex],hicutoff)
    ds1     = xr.open_dataset(ncname1).corr.load()
    ds_hp.append(ds1.copy())

#%% Do the Visualization

cints     = np.arange(-1,1.1,0.1)

fig,axs,_ = viz.init_orthomap(2,3,bboxplot,figsize=(18,8),)

for hp in range(2):
    
    if hp == 0:
        hpname = "Raw"
        dsin   = ds_raw
    else:
        hpname = "%02i-Month High Pass" % hicutoff
        dsin   = ds_hp
        
    
    for ex in range(3):
        
        ax      = axs[hp,ex]
        
        plotvar = dsin[ex].mean('ens').T#.isel(lags=lag,mons=kmonth).squeeze().mean('ens')
        # pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar.values.T,transform=proj,
        #                         vmin=-1,vmax=1,cmap='cmo.balance')
        pcm = ax.contourf(plotvar.lon,plotvar.lat,plotvar.values.T,transform=proj,levels=cints,cmap='cmo.balance')
        cl  = ax.contour(plotvar.lon,plotvar.lat,plotvar.values.T,transform=proj,levels=cints,colors='dimgray',
                         linewidths=0.55,)
        
        ax.clabel(cl,cints[::2])
        ax      = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
        
        #cb = viz.hcbar(pcm,ax=ax)
        
        if hp == 0:
            ax.set_title("%s" % (plotnames[ex]),fontsize=fsz_title)
        if ex == 0:
            viz.add_ylabel(hpname,ax=ax,fontsize=fsz_title)

cb = viz.hcbar(pcm,ax=axs.flatten(),fraction=0.045,pad=0.01)
cb.set_label("Instantaneous SST-SSS Cross-correlation (All Months)",fontsize=fsz_axis)

savename = "%sSST_SSS_Cross_Correlation_CESMvSM_hpf%02imon.png" % (figpath,hicutoff)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%%








