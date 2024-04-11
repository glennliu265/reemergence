#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Scrap script to recalculate and visualize spectra for regionally averaged
SST/SSS.

Copied upper section from from compare_regional_metrics.py

Created on Thu Apr 11 10:30:02 2024

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
import cartopy.crs as ccrs

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


#%% Indicate experiments to load

# Check updates after switching detrainment and correting Fprime (SST)
regionset       = "TCMPi24"
comparename     = "SST_AprilUpdate"
expnames        = ["SST_EOF_LbddEnsMean","SST_EOF_LbddCorr_Rerun","SST_CESM"]
expnames_long   = ["Stochastic Model (Exp Fit)","Stochastic Model (Corr.)","CESM1"]
expnames_short  = ["SM_old","SM_new","CESM"]
ecols           = ["forestgreen","goldenrod","k"]
els             = ["solid",'dashed','solid']
emarkers        = ["d","x","o"]

# # Check updates after switching detrainment and correting Fprime (SSS)
# regionset       = "TCMPi24"
# comparename     = "SSS_AprilUpdate"
# expnames        = ["SSS_EOF_Qek_LbddEnsMean","SSS_EOF_LbddCorr_Rerun","SSS_CESM"]
# expnames_long   = ["Stochastic Model (Exp Fit)","Stochastic Model (Corr.)","CESM1"]
# expnames_short  = ["SM_old","SM_new","CESM"]
# ecols           = ["forestgreen","goldenrod","k"]
# els             = ["solid",'dashed','solid']
# emarkers        = ["d","x","o"]

# regionset = "TCMPi24"
TCM_ver   = True # Set to just plot 2 panels

# Section between this copied from compare_regional_metrics ===================
#%% Load Regional Average SSTs and Metrics for the selected experiments

nexps = len(expnames)

seavar_all = []
var_all    = []
tsm_all   = []
rssts_all = []
acfs_all  = []
amv_all   = []
for e in range(nexps):
    
    # Get Experiment information
    expname        = expnames[e]
    
    if "SSS" in expname:
        varname = "SSS"
    elif "SST" in expname:
        varname = "SST"
    metrics_path    = output_path + expname + "/Metrics/"
    
    # Load Pointwise variance
    ds_std = xr.open_dataset(metrics_path+"Pointwise_Variance.nc").load()
    var_all.append(ds_std)
    
    # Load Seasonal variance
    ds_std2 = xr.open_dataset(metrics_path+"Pointwise_Variance_Seasonal.nc").load()
    seavar_all.append(ds_std2)
    
    # Load Regionally Averaged SSTs
    ds = xr.open_dataset(metrics_path+"Regional_Averages_%s.nc" % regionset).load()
    rssts_all.append(ds)
    
    # Load Regional Metrics
    ldz = np.load(metrics_path+"Regional_Averages_Metrics_%s.npz" % regionset,allow_pickle=True)
    tsm_all.append(ldz)
    
    # # Load Pointwise_ACFs
    # ds_acf = xr.open_dataset(metrics_path + "Pointwise_Autocorrelation_thresALL_lag00to60.nc")[varname].load()
    # acfs_all.append(ds_acf)  
    
    # # Load AMV Information
    # ds_amv = xr.open_dataset(metrics_path + "AMV_Patterns_SMPaper.nc").load()
    
"""

tsm_all [experiment][region_name].item()[metric]
where metric is :  one of ['acfs', 'specs', 'freqs', 'monvars', 'CCs', 'dofs', 'r1s']
see scm.compute_sm_metrics()

"""

#%% Load Mask
masknc = metrics_path + "Land_Ice_Coast_Mask.nc"
dsmask = xr.open_dataset(masknc).mask#__xarray_dataarray_variable__


#%% Get Region Information, Set Plotting Parameters

# Get Region Info
regions                     = ds.regions.values
bboxes                      = ds.bboxes.values
rdict                       = rparams.region_sets[regionset]
rcols                       = rdict['rcols']
rsty                        = rdict['rsty']
regions_long                = rdict['regions_long']
nregs                       = len(bboxes)

# Get latitude and longitude
lon = ds_acf.lon.values
lat = ds_acf.lat.values


#regions_long = ["Subpolar Gyre","Northern North Atlantic","Subtropical Gyre (East)","Subtropical Gyre (West)"]

# Plotting Information
bbplot                      = [-80,0,22,64]
mpl.rcParams['font.family'] = 'Avenir'
proj                        = ccrs.PlateCarree()
mons3                       = proc.get_monstr()

# Font Sizes
fsz_title                   = 20
fsz_ticks                   = 14
fsz_axis                    = 16
fsz_legend                  = 16

# End Section copied from compare_regional_metrics ============================
