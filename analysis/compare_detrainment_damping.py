#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compare detrending effect on detrainment damping (linear vs. ens mean)

Created on Wed Feb 28 10:36:58 2024

@author: gliu
"""


import time
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import xarray as xr
import sys
from tqdm import tqdm
import scipy as sp
import cartopy.crs as ccrs

import matplotlib as mpl

#%%

stormtrack = 0
if stormtrack:
    amvpath = "/home/glliu/00_Scripts/01_Projects/00_Commons/" # amv module
    scmpath = "/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/" # scm module
else:
    amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
    scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz

#%% Indicate the files

# compare_name = 'compareDT_TEMP_ens01'
# fns          = ["CESM1_HTR_FULL_lbd_d_params_TEMP_detrendlinear_lagmax3_ens01_regridNN.nc",
#        "CESM1_HTR_FULL_lbd_d_params_TEMP_detrendensmean_lagmax3_ens01_regridNN.nc"]
# fnnames      = ["Detrend (Linear)","Detrend (Ens. Mean)"]
#outpath      = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/ocn_var_3d/"
#mult = -1 # Use this if the data is raw, not flipped through preprocess_SSS Input

compare_name = 'compare_EnsAvg_SSS'
fns          = ["CESM1_HTR_FULL_SSS_Expfit_lbdd_monvar_detrendensmean_lagmax3_EnsAvg.nc",
                "CESM1_HTR_FULL_SSS_Expfit_lbdd_monvar_detrendensmean_lagmax3_Ens01.nc"]
fnnames      = ["Ens Avg.","Ens. 01"]
outpath      = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/damping/"
mult         = 1

figpath      = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20240301/"
proc.makedir(figpath)

def init_monplot():
    plotmon       = np.roll(np.arange(12),1)
    fig,axs,mdict = viz.init_orthomap(4,3,bboxplot=bboxplot,figsize=(18,18))
    for a,ax in enumerate(axs.flatten()):
        im = plotmon[a]
        ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
        ax.set_title(mons3[im],fontsize=fsz_axis)
    return fig,axs


#%% Load the files

nf = len(fns)
ds_all = []
for ff in range(nf):
    ncname=outpath + fns[ff]
    ds = xr.open_dataset(ncname).load()
    ds_all.append(ds)

#%% Visualize lbd_d for a particular month

# Plotting Params
mpl.rcParams['font.family'] = 'JetBrains Mono'
bboxplot                    = [-80,0,20,65]
proj                        = ccrs.PlateCarree()
lon                         = ds.lon.values
lat                         = ds.lat.values
mons3                       = proc.get_monstr()



vlms_base = [0,.3]
vlms_diff = [-.2,.2]
im        = 0

for im in range(12):
    cints_base     = [12,24,36,48,60,72]
    cints_diff     = [-12,-6,-3,0,3,6,12]
    
    
    fig,axs,mdict                = viz.init_orthomap(1,3,bboxplot,figsize=(14,8),constrained_layout=True,)
    for a,ax in enumerate(axs):
        
        ax                          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
        
        if a <2:
            plotvar = ds_all[a].lbd_d.isel(mon=im) * -mult
            cmap    = "inferno"
            title   = fnnames[a]
            vlms    = vlms_base
            cints   = cints_base
            linecol = "w"
            
        else:
            plotvar = (-ds_all[1].lbd_d.isel(mon=im) - -ds_all[0].lbd_d.isel(mon=im))
            cmap    = 'cmo.balance'
            title   =  "%s - %s" % (fnnames[1],fnnames[0])
            vlms    = vlms_diff
            cints   = cints_diff
            linecol = "k"
            
        ax.set_title(title)
        pcm     = ax.pcolormesh(lon,lat,plotvar,transform=proj,vmin=vlms[0],vmax=vlms[1],cmap=cmap)
        
        # Contours
        if a <2:
            cl      = ax.contour(lon,lat,1/plotvar,levels=cints,colors=linecol,linewidths=.55,transform=proj)
            ax.clabel(cl,fontsize=8)
            
        # Colorbar
        cb      = fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.02)
        cb.set_label("Detrainment Damping (1/mon)")
        
    plt.suptitle("%s. Detrainment Damping Comparison" % (mons3[im]),y=0.60,fontsize=16)
    
    savename = "%sDetrainment_Damping_Detrend_Comparison_%s_Ens01_mon%02i.png" % (figpath,compare_name,im+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Copy Above, but for 1 month



vlms_base = None#,[0,.3]
vlms_diff = None#,[-.2,.2]
im        = 0


cints_base     = [12,24,36,48,60,72]
cints_diff     = [-12,-6,-3,0,3,6,12]


fig,axs,mdict                = viz.init_orthomap(1,3,bboxplot,figsize=(14,8),constrained_layout=True,)
for a,ax in enumerate(axs):
    
    ax                          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
    
    if a <2:
        plotvar = ds_all[a].lbd_d.isel(mon=im) 
        cmap    = "inferno"
        title   = fnnames[a]
        vlms    = vlms_base
        cints   = cints_base
        linecol = "w"
        
    else:
        plotvar = (-ds_all[1].lbd_d.isel(mon=im) - -ds_all[0].lbd_d.isel(mon=im))
        cmap    = 'cmo.balance'
        title   =  "%s - %s" % (fnnames[1],fnnames[0])
        vlms    = vlms_diff
        cints   = cints_diff
        linecol = "k"
    
    if a == 0:
        vlms = [-.5,0]
    elif a == 1:
        vlms = [0,.2]
    
    ax.set_title(title)
    if vlms is None:
        pcm     = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=cmap)
    else:
        pcm     = ax.pcolormesh(lon,lat,plotvar,transform=proj,vmin=vlms[0],vmax=vlms[1],cmap=cmap)
    
    # Contours
    if a <2:
        cl      = ax.contour(lon,lat,1/plotvar,levels=cints,colors=linecol,linewidths=.55,transform=proj)
        ax.clabel(cl,fontsize=8)
        
    # Colorbar
    cb      = fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.02)
    cb.set_label("Detrainment Damping (1/mon)")
    
plt.suptitle("%s. Detrainment Damping Comparison" % (mons3[im]),y=0.60,fontsize=16)


