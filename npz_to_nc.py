#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Convert timescale files from npz to nc file for data exploration

Uses the top half of "viz_pointwise_autocorrelation.py"

Created on Mon May  2 19:42:03 2022

@author: gliu
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")

from amv import proc,viz
import scm
import numpy as np
import xarray as xr
from tqdm import tqdm 
import time
import cartopy.crs as ccrs

#%% Select dataset to postprocess

# Autocorrelation parameters
# --------------------------
lags        = np.arange(0,37)
lagname     = "lag%02ito%02i" % (lags[0],lags[-1]) 
thresholds  = [0,] # Standard Deviations
conf        = 0.95
tails       = 2

mconfig    = "HTR-FULL" # #"PIC-FULL"

thresholds = [0,]
thresname  = "thres" + "to".join(["%i" % i for i in thresholds])
varname    = "SSS" #"SST"

# Set Output Directory
# --------------------
figpath     = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20220502/'
proc.makedir(figpath)
outpath     = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/'
savename   = "%s%s_%s_autocorrelation_%s.npz" %  (outpath,mconfig,varname,thresname)
print("Loading the following dataset: %s" % savename)

# Plotting Params
# ---------------
colors   = ['b','r','k']
bboxplot = [-80,0,0,60]
mons3    = [viz.return_mon_label(m,nletters=3) for m in np.arange(1,13)]

#%% Set Paths for Input (need to update to generalize for variable name)

if mconfig == "SM": # Stochastic model
    # Postprocess Continuous SM  Run
    # ------------------------------
    datpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_output/"
    fnames      = ["forcingflxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0_1000yr_run2%02d_ampq0_method5_dmp0"%i for i in range(10)]
    mnames      = ["constant h","vary h","entraining"] 
elif "PIC" in mconfig:
    # Postproess Continuous CESM Run
    # ------------------------------
    datpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/"
    fnames     = ["CESM1_FULL_postprocessed_NAtl.nc","CESM1_SLAB_postprocessed_NAtl.nc"]
    mnames     = ["FULL","SLAB"] 
elif "HTR" in mconfig:
    # CESM1LE Historical Runs
    # ------------------------------
    datpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/CESM_proc/"
    fnames     = ["%s_FULL_HTR_lon-80to0_lat0to65_DTEnsAvg.nc" % varname,]
    mnames     = ["FULL",]
elif mconfig == "HadISST":
    # HadISST Data
    # ------------
    datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/"
    fnames  = ["HadISST_detrend2_startyr1870.npz",]
    mnames     = ["HadISST",]
elif mconfig == "ERSST":
    fnames  = ["ERSST_detrend2_startyr1900_endyr2016.npz"]
    
    
#%% Load in the data
st          = time.time()
ld          = np.load(savename,allow_pickle=True)
count_final = ld['class_count']
acs_final   = ld['acs'] # [lon x lat x (ens) x month x thres x lag]
lon         = ld['lon']
lat         = ld['lat']
thresholds  = ld['thresholds']
threslabs   = ld['threslabs']

nthres      = len(thresholds)
if "HTR" in mconfig:
    lens=True
    nlon,nlat,nens,nmon,_,nlags = acs_final.shape
else:
    lens=False
    nlon,nlat,nmon,_,nlags = acs_final.shape
print("Data loaded in %.2fs"% (time.time()-st))

#%% Place into a netCDF file

# Change to [thres x ens x lag x month x lat x lon]
acs_rs = acs_final.transpose(4,2,5,3,1,0)

coords_dict = {
               'thres':threslabs,
               'ens'  : np.arange(1,43,1),
               'lag'  : lags,
               'mon'  : np.arange(1,13,1),
               'lat'  : lat,
               'lon'  : lon,
               }

da = xr.DataArray(acs_rs,
            dims=coords_dict,
            coords=coords_dict,
            name = varname + "_ac"
            )