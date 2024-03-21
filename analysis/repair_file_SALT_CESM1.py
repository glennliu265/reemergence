#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Repeat SALT File for CESM1 at a single point.
Update 2024.03.20, TEMP file
Also converts z_t from cm to meters

Copied from Td_Sd_decay_vertical.py

Created on Wed Jan 24 21:31:01 2024

@author: gliu
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
from tqdm import tqdm
import scipy as sp

#%% Import Custom Modules
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl


#%% Set data paths

# Location
lonf           = 330
latf           = 50
locfn,loctitle = proc.make_locstring(lonf,latf)

datpath        = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/lon%s_lat%s/" % (lonf,latf)
figpath        = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20240322/"
proc.makedir(figpath)


outpath         = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/%s/" % locfn

#%% Load necessary files

vname  = "TEMP"
ncsalt = outpath + "CESM1_htr_%s.nc" % vname

ds_salt   = xr.open_dataset(ncsalt) 

z      = ds_salt.z_t.values/100
times  = ds_salt.time.values
salt   = ds_salt[vname].values # [Ens x Time x Depth ]
nens,ntime,nz = salt.shape

timesstr = ["%04i-%02i" % (t.year,t.month) for t in times]

ens = np.arange(nens)+1

#%% Quick Hovmuller (it appears that these are full anomalies)
# Note that Ens 32 seems to have a column of NaN values...

xtks = np.arange(0,ntime+1,120)
e      = 32
fig,ax = plt.subplots(1,1,figsize=(12,4))
pcm    = ax.pcolormesh(timesstr,z,salt[e,:,:].T,cmap="cmo.balance")
fig.colorbar(pcm)
ax.set_xticks(xtks)
ax.set_ylim(0,3000)
ax.invert_yaxis()
ax.set_title("%s in CESM1-PiControl (1920-2005)" % vname)

#%% Check for NaNs

# First, remove all depths where it is NaN
z_nan = np.where(np.nansum(salt,(0,1)) == 0.)[0] # Sum along Ens and Depth
print("The following %i depths have NaN values: \n\tDepths: %s, \n\tIndices: %s" % (len(z_nan),z[z_nan],z_nan))
print("Removing all values below depth %f" % (z[z_nan[0]]))
salt1 = salt[:,:,:z_nan[0]].copy()
z1    = z[:z_nan[0]]


#%% Second, check if there is an ensemble member that is all NaN
z_nan = np.where(np.isnan(np.sum(salt1,(1,2))))[0] # Sum along time and depth
print("Located NaN value at i=%i, e=%s" % (z_nan,ens[z_nan[0]]))
znan_ens = z_nan[0]
#sum_time_depth = np.sum(salt1,(1,2))
fig,ax=plt.subplots(1,1,figsize=(10,4))
pcm=ax.pcolormesh(timesstr,z1,salt1[z_nan[0],:,:].T)
fig.colorbar(pcm)
ax.set_xlabel("Time")
ax.set_ylabel("Depth")
ax.set_xlim([215,225])
ax.set_title("Ens %i, Location of NaN Value" % (ens[z_nan[0]]))

#print("Removing all values below depth %f" % (z[z_nan[0]]))
#%%
# Check if there is a timestep that is all NaN -------------------------
z_nan = np.where(np.isnan(np.sum(salt1,(0,2))))[0] # Sum along ens and depth
print("Located NaN value at i=%i, t=%s" % (z_nan,timesstr[z_nan[0]]))
znan_time = z_nan[0]
# Check Value
fig,ax=plt.subplots(1,1)
pcm=ax.pcolormesh(ens,z1,salt1[:,z_nan[0],:].T)
fig.colorbar(pcm)
ax.set_xlabel("Ens")
ax.set_ylabel("Depth")
ax.set_title("t=%s, Location of NaN Value" % (timesstr[z_nan[0]]))

#%% Linearly interpolate between the two values to Repair file

# If there is more than 1 bad point you can loop between this
val_before = salt1[znan_ens,znan_time-1,:]
val_0      = salt1[znan_ens,znan_time,:]
val_after  = salt1[znan_ens,znan_time+1,:]

x_in       = [0,2] # [2]
y_in       = np.array([val_before,val_after]) # [2,50]
interpfunc = sp.interpolate.interp1d(x_in,y_in,axis=0)
val_fill   = interpfunc([1]).squeeze()

# Check:
mult=0.01 # for offset
nz1 = val_after.shape[0]
fig,ax = plt.subplots(1,1,figsize=(4,12))
for zz in range(nz1):
    ax.plot([0,1,2],zz*mult+np.array([val_before[zz],val_fill[zz],val_after[zz]]),marker="o",label="level %i" % (zz))
ax.set_title("Linear Interp at each depth\n (Offsets to show values)")
ax.set_xlabel("Position (1=filled value")
ax.set_ylabel("Offset Value")
#%% Replace value

salt1[znan_ens,znan_time,:] = val_fill

#%% Replot to chek
xtks = np.arange(0,ntime+1,120)
e      = 32
fig,axs = plt.subplots(2,1,figsize=(16,6))

ax = axs[0]
pcm    = ax.pcolormesh(timesstr,z,salt[e,:,:].T,cmap="cmo.balance")
ax.set_title("Before")

ax = axs[1]
pcm    = ax.pcolormesh(timesstr,z1,salt1[e,:,:].T,cmap="cmo.balance")
ax.set_title("After")

ax.set_xticks(xtks)

for ax in axs:
    ax.set_ylim(0,3000)
    ax.invert_yaxis()
    ax.set_xticks(xtks)
    #ax.set_xlim([215,225])
    
plt.suptitle("%s in CESM1-PiControl (1920-2005)" % vname)

#%% Last check on existence of nans

print(np.any(np.isnan(salt1)))

#%% Save the output

newname = proc.addstrtoext(ncsalt,"_repaired",adjust=-1)
coords = {'ens':ens,
          'time':times,
          'z_t':z1,
          }
da_new = xr.DataArray(salt1,coords=coords,dims=coords,name=vname)
da_new.to_netcdf(newname,encoding={vname:{'zlib':True}})



# interpvals = []
# nz1 = val_after.shape[0]
# for z in range(nz1):
#     val_new = np.interp(1,[0,2],[])

