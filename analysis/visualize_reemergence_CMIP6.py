#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize Re-emergence in CMIP6

Created on Fri Feb  2 13:51:13 2024

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

#%% Import Custom Modules
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx


#%%

datpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CMIP6/proc/"
figpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/02_Figures/20240202/"
proc.makedir(figpath)


#%% Load ACFs

e       = 0
varname = "SSS"
nc      = "%sCESM2_%s_ACF_1850to2014_lag00to60_ens%02i.nc" % (datpath,varname,e+1)

ds  = xr.open_dataset(nc).isel(thres=2)[varname]
acf = ds.values # [lon x lat x month x lag]
lon = ds.lon.values
lat = ds.lat.values

T2 = np.sum(acf**2,3)

# DOESN'T SEEM TO BE WORKING?
# remidx_all = []
# for  kmonth in range(12):
#     reidx = proc.calc_remidx_simple(acf,kmonth,monthdim=-2,lagdim=-1)
#     remidx_all.append(reidx)
# remidx_all = np.array(remidx_all)

#%%
kmonth = 1

for kmonth in range(12):
    mons3 = proc.get_monstr(nletters=3)
    levels = np.arange(0,21,1)
    
    mpl.rcParams['font.family'] = 'JetBrains Mono'
    bboxplot                    = [-80,0,0,65]
    fig,ax                      = viz.geosubplots(1,1,figsize=(10,6))
    ax                          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="k")
    
    pcm = ax.contourf(lon,lat,T2[:,:,kmonth].T,cmap='cmo.dense',levels=levels)
    cl = ax.contour(lon,lat,T2[:,:,kmonth].T,colors='k',linewidths=0.8,linestyles='dotted',levels=levels)
    ax.set_title("CESM2 Salinity %s $T^2$ (Ens %02i) " % (mons3[kmonth],e+1))
    fig.colorbar(pcm,ax=ax,fraction=0.025,orientation='horizontal',pad=0.01,)
    
    savename = "%sCESM2_%s_T2_mon%02i_ens%02i.png" % (figpath,varname,kmonth+1,e+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Try for CESM1

cesm1ncs  = ['HTR-FULL_SST_autocorrelation_thres0.nc','HTR-FULL_SSS_autocorrelation_thres0.nc']
cesm1path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
vnames = ["SST","SSS"]

ds_opn = [xr.open_dataset(cesm1path+nc) for nc in cesm1ncs] # [thres x ens x lag x mon x lat x lon]
ds_opn = [ds.isel(thres=2) for ds in ds_opn]

ds  = ds_opn[0]
lon = ds.lon.values
lat = ds.lat.values
lag = ds.lag.values
ens = ds.ens.values
mon = ds.mon.values 

#%% Plot for a variable

ds = ds_opn[0]

vv = 0
vname = vnames[vv]

acf      = ds_opn[vv][vname].values # [Ens Lag Mon Lat Lon]
T2_cesm1 = np.sum(acf**2,1) 



#%%
# Plotting settings
bboxplot     = [-80,0,0,65]
mpl.rcParams['font.family'] = 'Avenir'
mons3 = proc.get_monstr(nletters=3)

fsz_tick=18
fsz_axis=20
#%%

e      = 0
kmonth = 0

for kmonth in range(12):
    
    levels = np.arange(0,21,1)
    
    #mpl.rcParams['font.family'] = 'JetBrains Mono'
    #bboxplot                    = [-80,0,0,65]
    fig,ax                      = viz.geosubplots(1,1,figsize=(10,6))
    ax                          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="k")
    plotvar = T2_cesm1[e,kmonth,:,:]
    pcm = ax.contourf(lon,lat,plotvar,cmap='cmo.dense',levels=levels)
    cl = ax.contour(lon,lat,plotvar,colors='k',linewidths=0.8,linestyles='dotted',levels=levels)
    ax.set_title("CESM1 %s %s $T^2$ (Ens %02i) " % (vname,mons3[kmonth],e+1))
    fig.colorbar(pcm,ax=ax,fraction=0.025,orientation='horizontal',pad=0.01,)
    
    savename = "%sCESM1_%s_T2_mon%02i_ens%02i.png" % (figpath,varname,kmonth+1,e+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Fancier Pot

kmonth = 0
e      = 0
varname=vname
lc = 'midnightblue'

levels = np.arange(0,21,1)

# Initialize Plot
fig,ax,mdict = viz.init_orthomap(1,1,bboxplot,figsize=(10,8.5),constrained_layout=True,)
ax           = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=20,
                                fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")

# PLot it
plotvar = T2_cesm1[e,kmonth,:,:]

pcm = ax.contourf(lon,lat,plotvar,cmap='cmo.dense',levels=levels,transform=mdict['noProj'])
cl = ax.contour(lon,lat,plotvar,colors='darkslategray',linewidths=.5,linestyles='solid',levels=levels,transform=mdict['noProj'])
ax.clabel(cl,levels[::2],fontsize=fsz_tick)
ax.set_title("CESM1 %s %s T$^2$ (Ens %02i) " % (vname,mons3[kmonth],e+1),fontsize=26)
cb = fig.colorbar(pcm,ax=ax,fraction=0.025,orientation='horizontal',pad=0.01,)#size=16)
cb.ax.tick_params(labelsize=fsz_tick)
cb.set_label("Timescale (Month)",fontsize=fsz_axis)
savename = "%sCESM1_%s_T2_mon%02i_ens%02i.png" % (figpath,varname,kmonth+1,e+1)
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)



