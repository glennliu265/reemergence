#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Postprocess Stochastic Model Output from SSS basinwide Integrations

Currently a working draft, will copy the essential functions once I have finalized things

Created on Wed Feb  7 17:23:00 2024

@author: gliu

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr
import sys
from tqdm import tqdm
import copy
import glob
import time

# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "Astraeus"

# First Load the Parameter File
sys.path.append("../")
import reemergence_params as rparams

# Paths and Load Modules
pathdict   = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])
sys.path.append(pathdict['scmpath'] + "../")
from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx
import stochmod_params as sparams

# Set needed paths
figpath     = pathdict['figpath']
input_path  = pathdict['input_path']
output_path = pathdict['output_path']
procpath    = pathdict['procpath']
rawpath     = pathdict['raw_path']

# Make Needed Paths
proc.makedir(figpath)

#%% Indicate Experiment Information

expname   = "SSS_EOF_NoLbdd"

if "SSS" in expname:
    varname = "SSS"
elif "SST" in expname:
    varname = "SST"
    
    
# Settings for CESM (assumes CESM output is located at rawpath)
anom_cesm = False                          # Set to false to anomalize CESM data
bbox_sim  = np.array([-80,   0,  20,  65]) # BBOX of stochastic model simulations, to crop CESM output

print("Performing Postprocessing for %s" % expname)
print("\tSearching for output in %s" % output_path)

#%% Load output (copied from analyze_basinwide_output_SSS)
# Takes 16.23s for the a standard stochastic model run (10 runs, 12k months)
print("Loading output...")
st          = time.time()

if "CESM" in expname:
    # Load NC files
    ncname    = "CESM1LE_%s_NAtl_19200101_20050101_bilinear.nc" % varname
    
    # Load DS
    ds_cesm   = xr.open_dataset(rawpath+ncname).squeeze()
    
    # Slice to region
    bbox_sim  = expdict['bbox_sim']
    ds_cesm   = proc.sel_region_xr(ds_cesm,bbox_sim)
    
    # Correct Start time
    ds_cesm   = proc.fix_febstart(ds_cesm)
    ds_cesm   = ds_cesm.sel(time=slice('1920-01-01','2005-12-31')).load()
    
    # Anomalize if necessary
    if anom_cesm is False:
        print("Detrending and deseasonalizing variable!")
        ds_cesm = proc.xrdeseason(ds_cesm) # Deseason
        ds_cesm = ds_cesm[varname] - ds_cesm[varname].mean('ensemble')
    else:
        ds_cesm = ds_cesm[varname]
    
    # Rename to "RUN" to fit the formatting
    ds_cesm = ds_cesm.rename(dict(ensemble='run'))
    ds_sm   = ds_cesm
    
else:
    
    # Load NC Files
    expdir      = output_path + expname + "/Output/"
    expmetrics  = output_path + expname + "/Metrics/"
    nclist      = glob.glob(expdir +"*.nc")
    nclist.sort()
    
    # Load DS, deseason and detrend to be sure
    ds_all      = xr.open_mfdataset(nclist,concat_dim="run",combine='nested').load()
    ds_sm       = proc.xrdeseason(ds_all[varname])
    ds_sm       = ds_sm - ds_sm.mean('run')
    
    # Load Param Dictionary
    dictpath    = output_path + expname + "/Input/expparams.npz"
    expdict     = np.load(dictpath,allow_pickle=True)
    
# Set Postprocess Output Path
metrics_path = output_path + expname + "/Metrics/" 


print("\tOutput loaded in %.2fs" % (time.time()-st))
print("\tMetrics will be saved to %s" % metrics_path)


# --- <> --- <> --- <> --- <> --- <> --- <> --- <> --- <> --- <> --- <> --- <> 
# Data Loading Complete... 
# --- <> --- <> --- <> --- <> --- <> --- <> --- <> --- <> --- <> --- <> --- <> 


# ----------------------------------------------------------------------
# %% 1) Compute Overall Pointwise Variance and Seasonal Average Variance
# ----------------------------------------------------------------------
print("Computing Pointwise Variances...")
st1            = time.time()

# Copy over the DataSet
ds             = ds_sm.copy()

# Compute Variances
dsvar_byens    = ds.std('time')
dsvar_seasonal = ds.groupby('time.season').std('time')

# Save output
# >> Save Overall Variance
savenamevar    = "%sPointwise_Variance.nc" % (metrics_path)#(run: 10, lat: 48, lon: 65)
edict          = {varname:{'zlib':True}}
dsvar_byens.to_netcdf(savenamevar,encoding=edict)

# >> Save Seasonal Variance
savenamevar    = "%sPointwise_Variance_Seasonal.nc" % (metrics_path)#(run: 10, lat: 48, lon: 65)
dsvar_seasonal.to_netcdf(savenamevar,encoding=edict)

print("\tSaved Pointwise and Seasonal Variances in %.2fs " % (time.time()-st1))

# ---------------------------------------------------
#%% Part (2) Regional Analysis
# ---------------------------------------------------
print("Computing Regional Averages...")
st2         = time.time()
ds          = ds_sm.copy()

#% Pull Parameters for regional analysis
bbxall      = sparams.bboxes
regionsall  = sparams.regions 
rcolsall    = sparams.bbcol

# Select Regions
regions_sel = ["SPG","NNAT","STGe","STGw"]
bboxes      = [bbxall[regionsall.index(r)] for r in regions_sel]
rcols       = [rcolsall[regionsall.index(r)] for r in regions_sel]

# Make an adjustment to exclude points blowing up (move from 65 to 60 N)
bboxes[0][-1] = 60
bboxes[1][-1] = 60


#%% Compute Regional Averages

# Calculate Regional Average Over each selected bounding box
ssts_reg = []
for r in range(len(regions_sel)):
    bbin      = bboxes[r]
    rsst      = proc.sel_region_xr(ds,bbin)     # (run: 10, time: 12000, lat: 22, lon: 37)
    ravg      = proc.area_avg_cosweight(rsst)   # (run: 10, time: 12000)
    
    ssts_reg.append(ravg)
ssts_reg      = xr.concat(ssts_reg,dim="r")     # (region: 4, run: 10, time: 12000)

# Place into new xr.DataArray
coords_new    = dict(regions=regions_sel,run=ssts_reg.run,time=ssts_reg.time)
coords_reg    = dict(regions=regions_sel,bounds=["W","E","S","N"])
da_rsst       = xr.DataArray(ssts_reg.values,coords=coords_new,dims=coords_new,name=varname)
da_reg        = xr.DataArray(np.array(bboxes),coords=coords_reg,dims=coords_reg,name='bboxes')
rsst_out      = xr.merge([da_rsst,da_reg])

# Save the Output
edict         = proc.make_encoding_dict(rsst_out)
savename_rsst = "%sRegional_Averages.nc" % (metrics_path) 
rsst_out.to_netcdf(savename_rsst,encoding=edict)
print("\tSaved Regional Averages in %.2fs" % (time.time() - st2))

# -------------------------------------
#%% Part (3) Compute Timeseries Metrics
# -------------------------------------
print("Computing Metrics for Regional Averages")
st3 = time.time()

# Set Metrics Options (Move this to the top)
nsmooth     = 150
lags        = np.arange(37)
pct         = 0.10
metrics_str = "nsmooth%03i_pct%03i_lagmax%02i" % (nsmooth,pct*100,lags[-1])

tsm_regs = {}
for r in tqdm(range(len(regions_sel))):
    
    
    rsst_in = ssts_reg.isel(r=r)
    rsst_in = np.where((np.abs(rsst_in)==np.inf) | np.isnan(rsst_in),0.,rsst_in)
    
    tsm = scm.compute_sm_metrics(rsst_in,nsmooth=nsmooth,lags=lags,pct=pct)
    tsm_regs[regions_sel[r]] = tsm
    
    
    
savename_tsm = "%sRegional_Averages_Metrics.npz" % (metrics_path) 
np.savez(savename_tsm,**tsm_regs,allow_pickle=True)
print("\n\tSaved Metrics for Regional Averages in %.2fs" % (time.time() - st3))


