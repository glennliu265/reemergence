#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Repair Ensemble Member 33 for SALT

Created on Wed Mar  6 13:43:50 2024

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
import scipy as sp

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

#%% Indicate path to file

# Get the file
vname    = "TEMP"#"SALT"
rawpath  = pathdict['raw_path'] + "ocn_var_3d/"
fn       = "%s_NATL_ens33.nc" % vname

ds       = xr.open_dataset(rawpath+fn)

tlon = ds.TLONG
tlat = ds.TLAT


#%% Find that stupid timestep

ntime,nz,nlat,nlon = ds[vname].shape

salt = ds[vname].load()


problemt = []
for t in tqdm(range(ntime)):
    
    # Check to see if everything is nan
    ts  = salt.isel(time=t)
    chk = np.all(np.isnan(ts))
    
    if chk:
        problemt.append(t)
        print("Found issue at timestep %i (%s)" % (t,ds.time.isel(time=t).values))
    
print(problemt)  
#%% Fix this using silly linear interpolation (taken from repair_file CESM1)

t = problemt[0]

def repair_timestep(ds,t):
    """Given a ds with a timestep with all NaNs, replace value with linear interpolation"""
    
    # Get steps before/after and dimensions
    val_before = ds.isel(time=t-1).values # [zz x tlat x tlon]
    val_0      = ds.isel(time=t).values   # [zz x tlat x tlon]
    val_after  = ds.isel(time=t+1).values # [zz x tlat x tlon]
    orishape   = val_before.shape
    newshape   = np.prod(orishape)
    
    # Do Linear Interp
    x_in       = [0,2]
    y_in       = np.array([val_before.flatten(),val_after.flatten()]) # [2 x otherdims]
    interpfunc = sp.interpolate.interp1d(x_in,y_in,axis=0)
    val_fill   = interpfunc([1]).squeeze()
    
    # Reshape and replace into repaired data array copy
    val_fill   = val_fill.reshape(orishape)
    #ds_fill   = xr.zeros_like(ds.isel(time=t))
    
    ds_new     = ds.copy()
    ds_new.loc[{'time':ds.time.isel(time=t)}] = val_fill
    return ds_new

salt_new = repair_timestep(salt,t)


# Check to make sure everything's ok
ds_out = ds.update(dict(SALT=salt_new))

problemt = []
for t in tqdm(range(ntime)):
    
    # Check to see if everything is nan
    ts  = ds_out.SALT.isel(time=t)
    chk = np.all(np.isnan(ts))
    
    if chk:
        problemt.append(t)
        print("Found issue at timestep %i (%s)" % (t,ds.time.isel(time=t).values))
if len(problemt) < 1:
    print("Everything is ok, it seems... (No NaN in repaired file.)")
savename_out = proc.addstrtoext(rawpath+fn,"_repaired",adjust=-1)
edict = proc.make_encoding_dict(ds_out)
ds_out.to_netcdf(savename_out,encoding=edict)

#%% Visualize repair
if debug:
    t = problemt[0]
    
    fig,axs = plt.subplots(1,2,subplot_kw={'projection':ccrs.PlateCarree()})
    
    for ii in range(2):
        ax = axs[ii]
        if ii == 0:
            
            ax.scatter(tlon,tlat,c=salt.isel(time=t,z_t=0))
            ax.set_title("Original")
        else:
            ax.scatter(tlon,tlat,c=salt_new.isel(time=t,z_t=0))
            ax.set_title("repair")
        
    plt.show()

    
    
    
    
    






