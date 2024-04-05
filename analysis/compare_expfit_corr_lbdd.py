#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:45:52 2024

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
machine = "stormtrack"

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

outpath     = procpath + "CESM1/NATL_proc/ocn_var_3d/"
#%% Load the files (corr method)

vname = "SALT"
nens  = 42

if vname == "TEMP":
    vname_surf = "SST"
elif vname == "SALT":
    vname_surf = "SSS"

# Indicate the number of netcdfs and ds
ncs    = []
ds_all = []
for e in range(nens):
    nc = "%sCESM1_HTR_FULL_corr_d_%s_detrendensmean_lagmax3_interp1_ceil0_imshift1_ens%02i_regridNN.nc" % (outpath,vname,e+1)
    ncs.append(nc)
    ds = xr.open_dataset(nc).lbd_d.load()   # {Mon x Lat x Lon}
    ds_all.append(ds)
ds_all = xr.concat(ds_all,dim='ens')        # {Ens x Mon x Lat x Lon}


#%% Load the files (detrainment computations)

# --------------------------------------------------------
# 1. Get Lbd_d estimates
ncname_lbdd = "damping/CESM1_HTR_FULL_%s_Expfit_lbdd_monvar_detrendensmean_lagmax3_EnsAll.nc" %  vname_surf
dsl         = xr.open_dataset(input_path+ncname_lbdd).lbd_d.load() # Ens x Mon x Lat x Lon

# --------------------------------------------------------
# 2. Compute delta_t from mixed-layer detrainment times
#    Note: More accurately, should load MLD cycle from each ensemble member...
ncname_h    = "mld/CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc"
dsh         = xr.open_dataset(input_path+ncname_h).h.load()

# <Resize DS> 
in_ds = [dsl,dsh]
out_ds = proc.resize_ds(in_ds)
dsl,dsh = out_ds

_,nlat,nlon=dsh.shape

kprev_all    = np.zeros(dsh.shape) * np.nan
delta_t_all = np.zeros(dsh.shape) * np.nan

for a in tqdm(range(nlat)):
    for o in range(nlon):
        
        # Find Detrainment Times at a point
        hpt = dsh.isel(lat=a,lon=o).values
        if np.any(np.isnan(hpt)):
            continue
        kprev,_ = scm.find_kprev(hpt)
        
        # Compute Difference in Months
        delta_t = np.arange(1,13) - kprev
        for tt in range(12):
            if delta_t[tt] <= 0: # For deepest layer and year crossings, add 12 months
                delta_t[tt] += 12
        delta_t[kprev==0.] = np.nan
        
        # Save variables
        kprev_all[:,a,o] = kprev.copy()
        delta_t_all[:,a,o] = delta_t.copy()
   
#Was going to change to DataArray but I guess not...
#coords = dict(mon=np.arange(1,13,1),lat=dsh.lat,lon=dsh.lon)
#ds_deltat = xr.DataArray()

# --------------------------------------------------------
# 3. Compute Correlation/Decay Factor
decay_factor = np.exp(-dsl * delta_t_all[None,:,:,:])


#%% First, let's check differences at a point
lonf = -30
latf = 50


expfitpt = decay_factor.sel(lon=lonf,lat=latf,method='nearest')
corrpt   = ds_all.sel(lon=lonf,lat=latf,method='nearest')


paramsin = [decay_factor,ds_all]
labels   = ["Exp. Fit","Correlation"]
cols     = ["k","royalblue"]



ptvals = [p.sel(lon=lonf,lat=latf,method='nearest') for p in paramsin]
#%% Plot the results

mons3  = proc.get_monstr()
fig,ax = viz.init_monplot(1,1,figsize=(12,4))

for ss in range(2):
    for e in range(nens):
        plotvar = ptvals[ss].isel(ens=e)
        ax.plot(mons3,plotvar,label="",alpha=0.1,c=cols[ss])
        
    mu    = ptvals[ss].mean('ens')
    sigma = ptvals[ss].std('ens')
    ax.plot(mons3,mu,label=labels[ss],c=cols[ss],marker="d")
    ax.fill_between(mons3,mu-sigma,mu+sigma,alpha=0.2,color=cols[ss])

ax.legend()
ax.set_ylabel("Correlation")
ax.set_title("Corr(Detrain,Entrain) for %s" % (vname))
plt.show()


#%%

# Plotting Params
mpl.rcParams['font.family'] = 'JetBrains Mono'
bboxplot                    = [-80,0,20,65]
proj                        = ccrs.PlateCarree()
lon                         = ds.lon.values
lat                         = ds.lat.values
mons3                       = proc.get_monstr()

plotmon                     = np.roll(np.arange(12),1)

fsz_title                   = 26
fsz_axis                    = 22
fsz_lbl                     = 10


#%% Plot Spatial Maps for a given month

fig,ax,_ = viz.init_orthomap()



