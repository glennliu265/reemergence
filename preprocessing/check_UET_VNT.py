#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Check UET and VNT Terms for the single point
Created on Thu Sep 26 12:20:00 2024

@author: gliu

"""


import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import scipy as sp

import matplotlib.pyplot as plt

import glob
import sys
import glob
import os

import tqdm
import time


#%% Import Custom Modules

# Import AMV Calculation
from amv import proc,viz
import amv.loaders as dl

# Import stochastic model scripts
import scm



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

proc.makedir(figpath)




#%% Load mean MLD and other variables

# Get Point Info
pointset    = "PaperDraft02"
ptdict      = rparams.point_sets[pointset]
ptcoords    = ptdict['bboxes']
ptnames     = ptdict['regions']
ptnames_long = ptdict['regions_long']
ptcols      = ptdict['rcols']
ptsty       = ptdict['rsty']

mldpath = input_path + "mld/"
ds_h    = xr.open_dataset(mldpath + "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc").load()



ds_hpt = [ds_h.h.sel(lon=ptcoords[ii][0],lat=ptcoords[ii][1],method='nearest') for ii in range(3)]


#%% Anomalize

def anomalize_lin(ds):
    
    ds = proc.fix_febstart(ds)
    ds = proc.xrdeseason(ds)
    
    coords = ds.dims
    #ds     = sp.signal.detrend(ds)
   # ds     = xr.DataArray(ds,coords=coords,dims=coords)
    
    return ds


#%% Load Data for points

rnames  = ["SAR","NAC","IRM",]
ds_pts  = []
dtmon   = 3600*24*30

for rr in range(3):
    
    searchstr = "%s*%s*.nc" % (rawpath,rnames[rr])
    ncfind    = glob.glob(searchstr)[0]
    ds = xr.open_dataset(ncfind).load()
    ds_pts.append(ds * dtmon)
    #print()
    
total_adv = [ds.UET + ds.VNT for ds in ds_pts]

#%% Examine vertical mean advection (for SST)

z_t = total_adv[0].z_t/100
fig,axs = plt.subplots(1,3,constrained_layout=True)

for ii in range(3):
    ax = axs[ii]
    ax.plot(total_adv[ii].mean('time'),z_t)
    ax.set_ylim([0,200])
    ax.invert_yaxis()
    
    ax.axhline(ds_hpt[ii].mean('mon'),color="k",ls='dashed')
    ax.set_xlim([-1,1])
    ax.set_title(rnames[ii])
    
    if ii == 1:
        ax.set_xlabel("Mean T Transport at Model Level ($\degree C$/month)")
    if ii == 0:
        ax.set_ylabel("Depth (meters)")

#%% Take Mean Along Mixed Layer Depth

z_t = total_adv[0].z_t/100


adv_zavg = np.zeros((3,1032)) * np.nan

indices  = np.arange(1032)
for ip in range(3):

    for im in range(12):
        monid = indices[im::12]
        
        mld_mon = ds_hpt[ip].isel(mon=im).item() # Convert to centimeters
        
        id_z    = proc.get_nearest(mld_mon,z_t.data) # Get Index of nearest mixed layer depth (inclusive)#z_t.sel(z_t=mld_mon,method='nearest')
        #print(monid)
        
        adv_in  = total_adv[ip].data[monid,:(id_z+1)].mean(1) #isel(time.dt.month=im)
        adv_zavg[ip,monid] = adv_in.copy()
        
        

coords        = dict(pt=rnames,time=total_adv[0].time)
adv_zavg      = xr.DataArray(adv_zavg,coords=coords,dims=coords,)
adv_zavg_anom = anomalize_lin(adv_zavg)



scycles       = proc.calc_clim(ds.data,dim=1)
#%% Now check the monthly variability


mons3         = proc.get_monstr()
monvar_zavg   = adv_zavg_anom.groupby('time.month').var('time')
fig,axs       = viz.init_monplot(1,3,figsize=(12,4))

for ip in range(3):
    plotvar = monvar_zavg.isel(pt=ip)
    ax = axs[ip]
    ax.plot(mons3,plotvar)


        # 
        





#%% Look at their interannual variability 

ds_monvar = [anomalize_lin(ds).groupby('time.month').var('time') for ds in ds_pts]



#%% Visualize Monthly Variance Contribution 

uet_ts = []
vnt_ts = []

for ii in range(3):
    
    
    
    
    