#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

General Script for calculating the re-emergence index,
given the pointwise ACFs for each month.

Works with output from:
    - [pointwise_autocorrelation_smoutput.py, ]
    - 

Created on Thu Jun 13 09:34:05 2024

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


#%% Declare Functions

def format_ds(ds):
    # acceptable dims: ens x mon x lag x lat x lon
    dimnames = list(ds.dims)
    if "mons" in dimnames:
        print("Rename mons ==> mon")
        ds = ds.rename(dict(mons='mon'))
    if "lags" in dimnames:
        print("Rename lags ==> lag")
        ds = ds.rename(dict(lags='lag'))
    if "run" in dimnames:
        print("Rename run ==> ens")
        ds = ds.rename(dict(run='ens'))
    ds = ds.squeeze()
    return ds
    
    

    # Set up function, for specified basemonth [kmonth]. Only return Re-emergence Index (REMIDX)
    calc_rei = lambda x: proc.calc_remidx_simple(x,kmonth,return_rei=True)
    
    # 
    rei_mon = xr.apply_ufunc(
        calc_rei,
        ds,
        input_core_dims=[['lag']],
        output_core_dims=[['lag']],
        vectorize=True,
        )

    
# def calc_remidx_xr(ac,minref=6,maxref=12,tolerance=3,debug=False,return_rei=False):
#     # Rewritten to work with just an input of acf [lags].
#     # For use case, see [calc_remidx_general.py]
#     # Rather than explicitly computing # years based on starting month, just assume 12 lags = 1 year
    
#     # Compute the number of years involved (month+lags)/12
#     # ex: if nlags = 37
#     #     if kmonth = 0, nyrs is just 37/12 = ~3 years (computes re-emergence for each year)
#     #     if kmonth = 11, nyrs is starts from Y1 (Dec) andgoes to (37+11)/12 (48), to Y4 Dec Re-emergence
#     nlags          = ac.shape[0]
#     nyrs           = int(np.floor((nlags) /12))
#     if debug:
#         maxids = []
#         minids = []
#     maxmincorr = np.zeros((2,nyrs,))  # [max/min x year]
#     for yr in range(nyrs):
        
#         # Get indices of target lags
#         minid = np.arange(minref-tolerance,minref+tolerance+1,1) + (yr*12)
#         maxid = np.arange(maxref-tolerance,maxref+tolerance+1,1) + (yr*12)
        
#         # Drop indices greater than max lag
#         maxlag = (ac.shape[0]-1)
#         minid  = minid[minid<=maxlag]
#         maxid  = maxid[maxid<=maxlag]
#         if len(minid) == 0:
#             continue
        
#         if debug:
#             print("For yr %i"% yr)
#             print("\t Min Ids are %i to %i" % (minid[0],minid[-1]))
#             print("\t Max Ids are %i to %i" % (maxid[0],maxid[-1]))
        
#         # Find minimum
#         mincorr  = np.min(np.take(ac,minid,axis=0),axis=0)
#         maxcorr  = np.max(np.take(ac,maxid,axis=0),axis=0)
        
#         maxmincorr[0,yr,...] = mincorr.copy()
#         maxmincorr[1,yr,...] = maxcorr.copy()
#         #remreidx[yr,...]     = (maxcorr - mincorr).copy()
        
#         if debug:
#             maxids.append(maxid)
#             minids.append(minid)
        
#         if debug:
#             ploti = 0
#             fig,ax = plt.subplots(1,1)
#             acplot=ac
#             #acplot = ac.reshape(np.concatenate([maxlag,],ac.shape[1:].prod()))
            
#             # Plot the autocorrelation function
#             ax.plot(np.arange(0,maxlag+1),acplot)
            
#             # Plot the search indices
#             ax.scatter(minid,acplot[minid],marker="x")
#             ax.scatter(maxid,acplot[maxid],marker="+")
            
            
#     if debug:
#         return maxmincorr,maxids,minids
#     if return_rei:
#         rei = maxmincorr[1,...] - maxmincorr[0,...]
#         return rei
#     return maxmincorr

#%% User Edits
# [lon x lat x mons x thres x lags]

# # SSS LbdE Rerun with Correct Sign
# vname   = 'SSS'
# expname = 'SSS_EOF_LbddCorr_Rerun_lbdE_neg'
# ncname  = "SM_%s_%s_autocorrelation_thresALL_lag00to60.nc" % (expname,vname)
# outpath = output_path + "%s/Metrics/" % expname
# outname = outpath + "REI_Pointwise.nc"

# SST Rerun
vname   = "SST"
expname = "SST_EOF_LbddCorr_Rerun"
ncname  = "SM_%s_%s_autocorrelation_thresALL_lag00to60.nc" % (expname,vname)
outpath = output_path + "%s/Metrics/" % expname
outname = outpath + "REI_Pointwise.nc"


# Try for CESM (SST)
vname   = "acf"
expname = "SST_CESM"
ncname  = "CESM1_1920to2005_SSTACF_lag00to60_ALL_ensALL.nc"
outpath = output_path + "%s/Metrics/" % expname
outname = outpath + "REI_Pointwise_new.nc"


# Try for CESM (SSS)
vname   = "acf"
expname = "SSS_CESM"
ncname  = "CESM1_1920to2005_SSSACF_lag00to60_ALL_ensALL.nc"
outpath = output_path + "%s/Metrics/" % expname
outname = outpath + "REI_Pointwise_new.nc"


#%%  Preprocess and Load DataArray with ACFs
st = time.time()
print("Loading data from: \n\t%s" % (procpath+ncname))
ds      = xr.open_dataset(procpath+ncname)
ds      = format_ds(ds)
ds      = ds[vname].load() # ('lon', 'lat', 'mon', 'lag')

# Get dimensions and positions
dimnames = list(ds.dims)
lagdim   = dimnames.index('lag')
monthdim = dimnames.index('mon') 

print("Data Loaded in %.2fs"  % (time.time()-st))

#%% Compute the ACF using xarray

st = time.time()

# Loop for each basemonth

# Set up function, only return Re-emergence Index (REMIDX)
calc_rei = lambda x: proc.calc_remidx_xr(x,return_rei=True)

# Apply looping through basemonth, lon, lat. ('lon', 'lat', 'mon', 'rem_year')
rei_mon = xr.apply_ufunc(
    calc_rei,
    ds,
    input_core_dims=[['lag']],
    output_core_dims=[['rem_year',]],
    vectorize=True,
    )

print("Function applied in in %.2fs" % (time.time()-st))

# Add numbering based on the re-emergence year
rei_mon['rem_year'] = np.arange(1,1+len(rei_mon.rem_year))

# Formatting to match output of [calc_remidx_CESM1]
rei_mon = rei_mon.rename({'rem_year':'yr'}) 
rei_mon = rei_mon.rename('rei')
if 'ens' in dimnames:
    rei_mon = rei_mon.transpose('mon','yr','ens','lat','lon')
else:
    rei_mon = rei_mon.transpose('mon','yr','lat','lon')
edict   = {'rei':{'zlib':True}}

# Save the output
rei_mon.to_netcdf(outname,encoding=edict)
print("Saved output to: \n\t%s" % (outname))

#%% For debugging, let's try to visualize things

proj            = ccrs.PlateCarree()
bboxplot        = [-80,0,20,65]

lon             = ds.lon.values
lat             = ds.lat.values


#%% Plot Re0emergence for a given month

kmonth                       = 1

rlevels                      = np.arange(0,0.55,0.05)
usepcm                       = False

fig,axs,mdict                = viz.init_orthomap(1,3,bboxplot,figsize=(20,8),constrained_layout=True,)


for yy in range(3):
    ax = axs[yy]
    ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
    
    plotvar = rei_mon.isel(yr=yy,mon=kmonth)
    if 'ens' in dimnames:
        plotvar = plotvar.mean('ens')
    
    ax.set_title("Year %02i" %  plotvar.yr.values,fontsize=20)
    
    if usepcm:
        pcm = ax.pcolormesh(lon,lat,plotvar,transform=proj,vmin=rlevels[0],vmax=rlevels[-1],cmap='cmo.dense')
    else:
        pcm = ax.contourf(lon,lat,plotvar,transform=proj,levels=rlevels,cmap='cmo.dense')
        cl  = ax.contour(lon,lat,plotvar,transform=proj,levels=rlevels,colors="dimgray",linewidths=0.55,)
        ax.clabel(cl,rlevels[::2])
    cb  = viz.hcbar(pcm,ax=ax,fraction=0.035)
    cb.set_label("REI Index",fontsize=14)
    

savename = '%sACF_REI_%s_%s_Y1toY3_pcm%i_mon%02i.png' % (figpath,expname,vname,usepcm,kmonth+1)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%%



