#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Copied from visualize_dz, quickly compute the depth correlation plots


Created on Wed May 22 18:15:42 2024

@author: gliu
"""


import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
from scipy import signal

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
machine = "stormtrack"

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


#%%




def calc_lagcovar_2d(ds3d,dspt,lags):
    #ds3d       = ds_in
    #dspt       = ds_in.isel(z_t=0)
    #lags       = np.arange(37)
    ntime,nother = ds3d.shape
    nyr = int(ntime/12)
    nlags = len(lags)
    # Separate to Mon x Year
    dsyrmon    = ds3d.values.reshape(nyr,12,nother).transpose(1,0,2) # [yr x mon x other] --> [mon x yr x other]
    dspt_yrmon = proc.year2mon(dspt.values)   # [mon x yr]
    
    ## Commented out this section so it can be done outside the function
    # # # # Deaseason
    # dsyrmon    = dsyrmon - np.mean(dsyrmon,1,keepdims=True)
    # dspt_yrmon = dspt_yrmon - np.mean(dspt_yrmon,1,keepdims=True)
    
    # # # # Detrend
    # dsyrmon         = signal.detrend(dsyrmon,axis=1,type='linear')
    # dspt_yrmon      = signal.detrend(dspt_yrmon,axis=1,type='linear')
    
    # Compute ACF for each month (using xarray)
    acf             = np.zeros((12,nother,nlags)) * np.nan
    for im in range(12):
        for kk in range(nother):
            tsbase = dspt_yrmon[:,:]
            tslag  = dsyrmon[:,:,kk]
            if np.any(np.isnan(tsbase)) or np.any(np.isnan(tslag)):
                continue
            
            ac = proc.calc_lagcovar(tsbase,tslag,lags,im+1,0,debug=False,)
            acf[im,kk] = ac.copy()
    return acf
    #coords = dict(mon=np.arange(1,13,1),z_t=ds3d.z_t)




#%% Load SST/TEMP data from above

locnames = ["Irminger","Labrador"]
# Load data
if machine != "stormtrack":
    outpath_rem = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/profile_analysis/"
else:
    outpath_rem = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/ptdata/profile_analysis/"
    
for ff in range(2):
    if ff == 0:
        ncname      = "IrmingerAllEns_SALT_TEMP.nc"
    elif ff ==1:
        ncname      = "LabradorAllEns_SALT_TEMP.nc"
    
    ds          = xr.open_dataset(outpath_rem + ncname).load()
    
    # Demean and Deseason, Fix February Start
    dsvars = [ds.TEMP,ds.SALT]
    vnames = ["TEMP","SALT"]
    dsanoms = [ds - ds.mean('ensemble') for ds in dsvars]
    dsanoms = [proc.xrdeseason(ds) for ds in dsanoms]
    dsanoms = [proc.fix_febstart(ds) for ds in dsanoms]
    
    lags = np.arange(37)
    
    for di in range(5):
        # if di == 0 and ff == 0:
        #     print("Skipping center point for Irminger Basin (can manually delete this)...")
        #     continue
        
        rem_byvar = []
        for vv in range(2):
            
            rem_byens = []
            
            for e in tqdm.tqdm(range(42)):
                
                ds_in      = dsanoms[vv].isel(dir=di).isel(ensemble=e)
                ds3d       = ds_in
                dspt       = ds_in.isel(z_t=0)
                ac         = calc_lagcovar_2d(ds3d,dspt,lags)
                
                rem_byens.append(ac)
            
            rem_byens = np.array(rem_byens)
            rem_byvar.append(rem_byens)
                
        coords      = dict(ens=np.arange(1,43,1),mon=np.arange(1,13,1),z_t=ds3d.z_t,lag=lags,)
        rem_byvar   = [xr.DataArray(rem_byvar[ii],coords=coords,dims=coords,name=vnames[ii]) for ii in range(2)]
         
        ds_out      = xr.merge(rem_byvar)
        outname     = "%s%sAllEns_SALT_TEMP_REM_3DCorr_%s.nc" % (outpath_rem,locnames[ff],ds.dir[di].values.item())
        edict       = proc.make_encoding_dict(ds_out)
        ds_out.to_netcdf(outname,encoding=edict)


#%% Visualize the output (written locally for Astraeus)





#%% Might move this section to [viz_icefrac]

vnames  = ["TEMP","SALT"]
ncname  = "LabradorAllEns_SALT_TEMP_REM_3DCorr_Center.nc"
ncpath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/profile_analysis/"
dsrem   = xr.open_dataset(ncpath+ncname).load()


# Plotting Selections
kmonth   = 1
mons3    = proc.get_monstr()
cints    = np.arange(-1,1.05,0.05)
xtks     = np.arange(0,37,3)
fig,axs  = plt.subplots(2,1,figsize=(12,6.5))

for vv in range(2):
    ax = axs[vv]
    # Plot Temp
    plotvar = dsrem.isel(mon=kmonth).mean('ens')[vnames[vv]]
    cf = ax.contourf(dsrem.lag,dsrem.z_t/100,plotvar,levels=cints,cmap='RdBu_r')

# # Plot Salt
# plotvar = dsrem.isel(mon=1).mean('ens').SALT
# cl = ax.contour(dsrem.lag,dsrem.z_t/100,plotvar,levels=cints,colors="k",lw=0.75)
# ax.clabel(cl)

    ax.invert_yaxis()
    ax.set_xticks(xtks)
    ax.set_ylabel("Depth (m)")
    ax.set_xlabel("Lag (months)")

    cb = fig.colorbar(cf,ax=ax,pad=0.01,fraction=0.02)
#plt.show()





