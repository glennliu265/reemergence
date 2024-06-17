#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Copied upper section of visualize_dz

Compute the vertical gradient at the detrainment depth
Also compute the second derivative (for diffusion)

Created on Tue May 21 10:41:52 2024

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

#%% Other INputs

dz2      = True # Set to True to take the 2nd derivative, False for first.


vnames   = ["TEMP","SALT"]
vcolors  = ["hotpink","navy"]
vunits   = ["$\degree C$","$psu$"]
vmarkers = ['o','x']
bbox_reg = [-80,0,20,65]

#%% Work with data postprocessed by [viz_icefrac.py]

# Path to 3D datasets
dpath       = rawpath + "ocn_var_3d/"
outpath     = dpath

# Load detrainment data
mpath       = input_path + "mld/"
ds_hdetrain = xr.open_dataset(mpath + "CESM1_HTR_FULL_hdetrain_NAtl.nc").load()
ds_hdetrain = proc.sel_region_xr(ds_hdetrain,bbox_reg)
lat         = ds_hdetrain.lat
lon         = ds_hdetrain.lon
nlon,nlat=len(lon),len(lat)


# Loop for ens, variable
vv = 1
e  = 1

# Load the netCDF
for vv in range(2):
    vname= vnames[vv]
    
    
    detrain_gradients = np.zeros((42,12,12,nlat,nlon)) * np.nan # [Ens,Entrain Mon, Mon, Lat, Lon]
    for e in range(42):
        
        ncname = "%s%s_NATL_ens%02i.nc" % (dpath,vname,e+1)
        if e == 32:
            ncname = proc.addstrtoext(ncname,"_repaired",adjust=-1)
        ds = xr.open_dataset(ncname).load() # ('time', 'z_t', 'nlat', 'nlon')
        
        # Compute seasonal cycle
        ds_scycle = ds[vname].groupby('time.month').mean('time') #  ('month', 'z_t', 'nlat', 'nlon')
        
        # Compute centered difference
        #invar  = ds_scycle.values
        #z_t    = ds_scycle.z_t.values/100 # Convert cm --> m
        ds_scycle['z_t'] = ds.z_t.values/100
        dxdz   = ds_scycle.differentiate('z_t')#np.gradient(invar,z_t,axis=1) # Compute Gradient over Z axis 
        
        if dz2: # Take centered difference (again)
            print("Taking second derivative")
            dxdz   = dxdz.differentiate('z_t')
            
        
        # Looping for each point
        for o in tqdm.tqdm(range(nlon)):
            for a in range(nlat):
                
                # Retrieve detrainment depth and gradients at the point
                hdet_pt = ds_hdetrain.h.isel(ens=e,lat=a,lon=o) # mon
                lonf=hdet_pt.lon.values.item()
                latf=hdet_pt.lat.values.item()
                if lonf<0:
                    lonf = lonf+360
                dxdz_pt = proc.find_tlatlon(dxdz,lonf,latf,verbose=False) # (month: 12, z_t: 44)
                
                # For each entraining month
                for im in range(12):
                    # Get Detraining Depth
                    
                    hdet = hdet_pt[im].values.item()
                    
                    if np.isnan(hdet):
                        continue
                            
                    # Select nearest gradient
                    dxdz_mon = dxdz_pt.sel(z_t=hdet,method='nearest').values
                    detrain_gradients[e,im,:,a,o] = dxdz_mon.copy()
            
    coords=dict(ens=np.arange(1,43,1),
        entrain_mon=np.arange(1,13,1),
        mon=np.arange(1,13,1),
        lat=lat,
        lon=lon,
        )
    da_out  = xr.DataArray(detrain_gradients,coords=coords,dims=coords,name='grad')
    edict   = {'grad':{'zlib':True}}
    if dz2:
        outname = "%sCESM1_HTR_%s_Detrain_Gradients_dz2.nc" % (outpath,vnames[vv])
    else:
        outname = "%sCESM1_HTR_%s_Detrain_Gradients.nc" % (outpath,vnames[vv])
    da_out.to_netcdf(outname,encoding=edict,)

#%% New section (compute the local standard deviation, for normalization purposes)

for vv in range(2):
    vname = vnames[vv]
    detrain_stdevs = np.zeros((42,12,12,nlat,nlon)) * np.nan # [Ens,Entrain Mon, Mon, Lat, Lon]
    
    for e in range(42):
        
        ncname = "%s%s_NATL_ens%02i.nc" % (dpath,vname,e+1)
        if e == 32:
            ncname = proc.addstrtoext(ncname,"_repaired",adjust=-1)
        ds = xr.open_dataset(ncname).load() # ('time', 'z_t', 'nlat', 'nlon')
        
        # Compute seasonal cycle
        # ds_scycle = ds[vname].groupby('time.month').mean('time') #  ('month', 'z_t', 'nlat', 'nlon')
        # ds
        # # Compute centered difference
        # #invar  = ds_scycle.values
        # #z_t    = ds_scycle.z_t.values/100 # Convert cm --> m
        # ds_scycle['z_t'] = ds.z_t.values/100
        # dxdz   = ds_scycle.differentiate('z_t')#np.gradient(invar,z_t,axis=1) # Compute Gradient over Z axis 
        
        # Looping for each point
        for o in tqdm.tqdm(range(nlon)):
            for a in range(nlat):
                
                # Retrieve detrainment depth and gradients at the point
                hdet_pt = ds_hdetrain.h.isel(ens=e,lat=a,lon=o) # mon
                lonf    = hdet_pt.lon.values.item()
                latf    = hdet_pt.lat.values.item()
                if lonf<0:
                    lonf = lonf+360
                dspt = proc.find_tlatlon(ds[vname],lonf,latf,verbose=False) # (month: 12, z_t: 44)
                
                
                
                # For each entraining month
                for im in range(12):
                    
                    
                    
                    # Get Detraining Depth
                    
                    hdet = hdet_pt[im].values.item()
                    
                    if np.isnan(hdet):
                        continue
                            
                    # Select nearest gradient
                    dspt_z                      = dspt.sel(z_t=hdet,method='nearest')
                    dspt_stdmon                 = dspt_z.groupby('time.month').std('time')
                    detrain_stdevs[e,im,:,a,o]  = dspt_stdmon.values
    
    coords=dict(ens=np.arange(1,43,1),
        entrain_mon=np.arange(1,13,1),
        mon=np.arange(1,13,1),
        lat=lat,
        lon=lon,
        )
    
    da_out  = xr.DataArray(detrain_stdevs,coords=coords,dims=coords,name='std')
    edict   = {'std':{'zlib':True}}
    outname = "%sCESM1_HTR_%s_Detrain_Stdev.nc" % (outpath,vnames[vv])
    da_out.to_netcdf(outname,encoding=edict,)
    print("Saved to %s"% outname)






# -----------------------------------
#%% Compute surface variance
# -----------------------------------

# Load TEMP extracted with 


#%% ---------------------------------------------------------------------------
# Script ends here, below was a scrap section focused on the Irminger/Lab sea profile analysis

#%%

# Load the data
outpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/profile_analysis/"
ncname  = "IrmingerAllEns_SALT_TEMP.nc"
ds      = xr.open_dataset(outpath+ncname).load()

# Load the mixed layer depth
mldpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/mld/"
mldnc   = "CESM1_HTR_FULL_HMXL_NAtl.nc"
ds_mld  = xr.open_dataset(mldpath + mldnc).load()

# Load detrainment depth computed from [calc_detrainment_depth]
hdtnc = "CESM1_HTR_FULL_hdetrain_NAtl.nc"
ds_hdetrain = xr.open_dataset(mldpath+hdtnc).load()

# Load estimates of SALT and TEMP


vnames  = ["TEMP","SALT"]
vcolors = ["hotpink","navy"]
vunits  = ["$\degree C$","$psu$"]
vmarkers = ['o','x']