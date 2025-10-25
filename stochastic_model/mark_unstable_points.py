#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Mark Unstable Points

 - Copied sections from /smio/scrap/investigate_blowup

Identify which points blew up during the stochastic model simulation

Created on Fri Oct 24 14:30:07 2025

@author: gliu

"""

import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import scipy as sp

import matplotlib.patheffects as PathEffects

import matplotlib.pyplot as plt
import sys
import glob
import os

import tqdm
import time

#%%

amvpath = "/home/glliu/00_Scripts/01_Projects/00_Commons/" # amv module
scmpath = "/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/" # scm module

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl

#%% User Edits: Set up Region Average Directory

# Indicate Path to Area-Average Files
dpath_aavg      = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/05_SMIO/data/region_average/"
regname         = "SPGNE"
bbsel           = [-40,-15,52,62]

# Set up Output Path
locfn,locstring = proc.make_locstring_bbox(bbsel,lon360=True)
bbfn            = "%s_%s" % (regname,locfn)
outpath         = "%s%s/" % (dpath_aavg,bbfn)
proc.makedir(outpath)

#%% User Edits

# outformat = "{outpath}{expname}_{vname}_{ystart:04d}_{yend:04d}_{procstring}.nc"
# outname    = outformat.format(outpath=outpath,
#                               expname=expname,
#                               vname=vname,
#                               ystart=ystart,
#                               yend=yend,
#                               procstring=procstring)#"CESM2_POM3_SHF_0200"

outformat       = "%s%s_%s_%04d_%04d_%s.nc"

# If smoutput is <True>... ----------------------------------------------------
# Use sm loader and output path to metrics folder
expname         = "SST_ORAS5_avg_GMSST_EOFmon_usevar_NATL" #"SST_ORAS5_avg_EOF" #"SST_ORAS5_avg_mld003" #"SST_ORAS5_avg" #"SST_ERA5_1979_2024"
vname           = "SST"
concat_dim      = "time"

sm_output_path  = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/sm_experiments/"
outpath         = "%s%s/Metrics/" % (sm_output_path,expname) # Save into experiment directory



#%% Load the Output
nclist = dl.load_smoutput(expname,output_path=sm_output_path,return_nclist=True)#return_nclist=True)

#%% 

nruns = len(nclist)
for rr in range(nruns):
    
    ds = xr.open_dataset(nclist[rr])
    
    # Make the dummy variable
    if rr == 0:
        ntime,nlat,nlon = ds.SST.shape
        blowup_index    = np.zeros((nruns,nlat,nlon)) * np.nan
    
    for a in tqdm.tqdm(range(nlat)):
        for o in tqdm.tqdm(range(nlon)):
            dspt   = ds.SST.isel(lat=a,lon=o)
            nanchk = np.where(np.isnan(dspt))[0]
            if len(nanchk) == 0:
                continue
            else:
                #break
                if np.all(np.isnan(dspt)): # Just a Masked out Point (all NaNs)
                    continue
                else:
                    firstnan = nanchk[0]
                    blowup_index[rr,a,o] = firstnan
    #dsnan = ds.SST.where(np.isnan(ds.SST))
    
coords  = dict(run=np.arange(nruns),lat=ds.SST.lat,lon=ds.SST.lon)
dsout   = xr.DataArray(blowup_index,coords=coords,dims=coords,name="nanid")
outname = "%sBlowup_Points.nc" % outpath
edict   = proc.make_encoding_dict(dsout)
dsout.to_netcdf(outname,encoding=edict)

#%%

