#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize EOF Forcing (Evaporation, Precipitation, Stochastic Heat Flux, Etc)

Created on Thu Feb 29 14:42:40 2024

@author: gliu
"""

import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs

#%% Import Custom Modules
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx

#%% Load some files


figpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20240301/"
rawpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"
input_path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/forcing/"
proc.makedir(figpath)

vname   = "SSS"

if vname == "SSS":
    vunits = "psu/mon"
elif vname == "SST":
    vunits = "W/m2"
    
dtplot = 3600*24*30

#%% Load the forcing files (copied from viz_Qek_output.py)


# Load Output of EOF Analysis
nceof        = "EOF_Monthly_NAO_EAP_Fprime_nomasklag1_nroll0_NAtl.nc"
dseof        = xr.open_dataset(rawpath+nceof).load()
varexp       = dseof.varexp

# Load NAO-related tau 
savename     = "%sCESM1_HTR_FULL_Monthly_TAU_NAO_nomasklag1_nroll0.nc" % (rawpath)
nao_taus     = xr.open_dataset(savename)
dstaux       = nao_taus.TAUX
dstauy       = nao_taus.TAUY

# Load Precip and Evaporation Forcing
sss_forcings = ["CESM1_HTR_FULL_LHFLX_EOF_nomasklag1_nroll0_NAtl.nc","CESM1_HTR_FULL_PRECTOT_EOF_nomasklag1_nroll0_NAtl.nc"]
ds_forcings  = [xr.open_dataset(input_path+fn).load() for fn in sss_forcings]
dsevap,dsprec = ds_forcings


#%% Set plotting parameters

# Plotting Params
mpl.rcParams['font.family'] = 'JetBrains Mono'
bboxplot                    = [-80,0,20,65]
proj                        = ccrs.PlateCarree()
lon                         = dstaux.lon.values
lat                         = dstaux.lat.values
mons3                       = proc.get_monstr()

vnames      = ["LHFLX","PRECTOT"]
vnames_long = (r"Evaporation (W/m2)",r"Precipitation (m/s)")
vcmaps      = ["cmo.balance","cmo.tarn"]


vcmax = [25,2e-8]

#vnames

#%% Visualize Different Modes

fsz_axis  = 16
fsz_title = 20
im  = 1
e   = 0

xint=4
yint=4
tauscale = 0.8
    
    
for im in range(12):
    

    
    
    fig,axs,mdict=viz.init_orthomap(2,4,figsize=(16,6.5),bboxplot=bboxplot)
    
    for v in range(2):
        for a in range(4):
            ax = axs[v,a]
            ax = viz.add_coast_grid(ax,bboxplot,fill_color='lightgray')
        
            # Labeling
            if v == 0:
                ax.set_title("Mode %i (%.2f"  % (a+1,varexp.isel(ens=e,mode=a,mon=im).values*100)+ "%)",
                             fontsize=fsz_axis)
            if a == 0:
                viz.add_ylabel(vnames_long[v],ax=ax,fontsize=fsz_axis)
            
            # Plot Evap/Precip
            plotvar = ds_forcings[v][vnames[v]].isel(mode=a,mon=im,ens=e)
            pcm     = ax.pcolormesh(lon,lat,plotvar,transform=proj,cmap=vcmaps[v],vmin=-vcmax[v],vmax=vcmax[v])
    
        
            # Plot Wind Stress
            x    = dstaux.lon.values[::xint]
            y    = dstaux.lat.values[::yint]
            taux = dstaux.isel(mon=im,mode=a,ens=e).values[::yint,::xint]
            tauy = dstauy.isel(mon=im,mode=a,ens=e).values[::yint,::xint]
            qv2  = ax.quiver(x,y,taux*-1,tauy*-1,scale=tauscale,color="gray",transform=proj)
            
            #fig.colorbar(pcm,ax=ax,fraction=0.015,pad=0.01,orientation='horizontal')
        fig.colorbar(pcm,ax=axs[v,:].flatten(),fraction=0.015,pad=0.01,orientation='vertical')
    plt.suptitle("EOF Pattern for Month %s, Ens %02i" % (mons3[im],e+1),fontsize=fsz_title)
    
    savename = "%sEvapPrecipEOFPattern_Mon%02i_Ens%02i.png" % (figpath,im+1,e+1)
    plt.savefig(savename,dpi=150,bbox_inches="tight")
    
    
    




