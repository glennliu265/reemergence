#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Plot to quickly visualize profiles at given points 
(to get a sense of how MLD is determined by CESM1)

Do this for 1 ensemble member

NOTE: Reconcile this script with [depth_analysis/visualize_dz.py] which does the
same thing but for a single point

Copied upper section (and various parts) for viz_icefrac
Using

Created on Tue Jun 11 13:59:24 2024

@author: gliu
"""
import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs

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

#%% User edits and plotting

vnames      = ["TEMP","SALT"]
vcolors     = ["hotpink","navy"]
vunits      = ["$\degree C$","$psu$"]
vmarkers    = ['o','x']


#%% Load TEMP and SALT for 1 ensemble member

st = time.time()

ds_3d = []
for vv in range(2):
    dp = rawpath + "ocn_var_3d/"
    ncname = "%s%s_NATL_ens01.nc" % (dp,vnames[vv])
    ds = xr.open_dataset(ncname).load()
    ds_3d.append(ds)

#% Load Mixed-Layer Depth

ds_hmxl = xr.open_dataset(rawpath + "CESM1LE_HMXL_NAtl_19200101_20050101_bilinear.nc").isel(ensemble=0).load()

print("Loaded data in %.2fs" % (time.time()-st))

#%% Plot Profiles for a point

lonf    = -38
latf    = 62
lonf360 = lonf + 360


locfn,loctitle = proc.make_locstring(lonf,latf)

#%% Select data for a point

dspt = [proc.find_tlatlon(ds,lonf360,latf,) for ds in ds_3d]
hpt  = ds_hmxl.sel(lon=lonf,lat=latf,method='nearest')/100 # Convert to meters


z_t = dspt[0].z_t/100
times = proc.cftime2str(dspt[0].TEMP.time.values)

#%% Plot some profiles for random timesteps

t = 22

fig,ax=plt.subplots(1,1,constrained_layout=True,figsize=(4,8))


# Make additional axes
ax2 = ax.twiny()
axs = [ax,ax2]

# Plot the profile
lns = []
for vv in range(2):
    
    axin = axs[vv]
    
    plotvar = dspt[vv].isel(time=t)[vnames[vv]].values
    #tstr   
    #plotvar     = dspt[vnames[vv]].groupby('time.month').mean('time').T.isel(month=im)#values 
    l, = axin.plot(plotvar,z_t,c=vcolors[vv],marker=vmarkers[vv],label=vnames[vv])
    axin.set_xlabel("%s [%s]" % (vnames[vv],vunits[vv]),c=vcolors[vv])
    axin.tick_params(axis='x', colors=vcolors[vv])
    
    lns.append(l)

# Adjust the Axis Colors and Orientation
plt.gca().invert_yaxis()
# Adjust Axis Colors
ax2.spines['bottom'].set_color(vcolors[0])
ax2.spines['top'].set_color(vcolors[1])

# Plot the MLD
hl = ax.axhline([hpt.HMXL.isel(time=t)],lw=0.75,label="Mixed Layer Depth",c='k')
lns.append(hl)

ax.set_title("Profile @ %s\nt = %s" % (loctitle,times[t]))


labs = [l.get_label() for l in lns]
ax.legend(lns,labs,loc="lower center")

#%% Check YO's Hypothesis

#im = 1


vvskip = 1

chooset = [0,0+12]
lst     = ['dotted','dashed']

fig,ax=plt.subplots(1,1,constrained_layout=True,figsize=(4,8))


# Make additional axes
ax2 = ax.twiny()
axs = [ax,ax2]
lns = []

# Plot the profile
profiles_byvar = [] # [t][TEMP,SALT]
mlds           = []
for nt in range(len(chooset)):
    t = chooset[nt]
    #lns = []
    
    # Plot the Profiles
    profiles = []
    for vv in range(2):
        if vv == vvskip:
            profiles.append(None)
            continue
        
        axin = axs[vv]
        
        plotvar = dspt[vv].isel(time=t)[vnames[vv]].values
        #tstr   
        #plotvar     = dspt[vnames[vv]].groupby('time.month').mean('time').T.isel(month=im)#values 
        l, = axin.plot(plotvar,z_t,c=vcolors[vv],ls=lst[nt],label=vnames[vv] + " (t=%s)" % times[t])
        lns.append(l)
        axin.set_xlabel("%s [%s]" % (vnames[vv],vunits[vv]),c=vcolors[vv])
        axin.tick_params(axis='x', colors=vcolors[vv])
        
        profiles.append(plotvar)
        
    # Plot the MLD
    hplot   = hpt.HMXL.isel(time=t)
    hl      = ax.axhline([hplot],lw=2,label="MLD = %.2f (t=%s)" % (hplot,times[t]),c='k',ls=lst[nt])
    lns.append(hl)
    mlds.append(hplot)
    
    profiles_byvar.append(profiles)
    
    #lns.append(l)
    
# Plot Mean Profiles and MLD

# Plot the Profiles

for vv in range(2):
    
    if vv == vvskip:
        continue
    
    axin = axs[vv]
    
    plotvar = (profiles_byvar[0][vv] + profiles_byvar[1][vv])/2
    #tstr   
    #plotvar     = dspt[vnames[vv]].groupby('time.month').mean('time').T.isel(month=im)#values 
    l, = axin.plot(plotvar,z_t,c=vcolors[vv],ls="solid",label=vnames[vv] + " (mean)")
    lns.append(l)
    axin.set_xlabel("%s [%s]" % (vnames[vv],vunits[vv]),c=vcolors[vv])
    axin.tick_params(axis='x', colors=vcolors[vv])

    
# Plot the MLD
ploth = (mlds[0]  + mlds[1])/2
hl = ax.axhline([ploth],lw=2,label="MLD = %.2f (Mean)" % (ploth),c='k',ls='solid')
lns.append(hl)


# Adjust the Axis Colors and Orientation
plt.gca().invert_yaxis()
# Adjust Axis Colors
ax2.spines['bottom'].set_color(vcolors[0])
ax2.spines['top'].set_color(vcolors[1])

ax.set_title("%s Profile @ %s" % (vnames[vv-1],loctitle))


if vvskip == 1:
    ax.set_xlim([6.25,7.25])
    ax.grid(True,ls='dashed',c='gray')
labs = [l.get_label() for l in lns]
ax.legend(lns,labs,loc="lower right")
# Initialize Plot

#%% For a given month, plot 

