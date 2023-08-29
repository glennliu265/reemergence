#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute Salinity-Heat Flux Feedback


Takes SSS regridded by [stochmod/prep_mld_PIC]

Works with FSNS, FLNS, 

Takes raw TS, LANDFRAC, ICEFRAC

Created on Tue Aug 29 13:52:56 2023

@author: gliu
"""


import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import glob
import sys
from tqdm import tqdm

#%% Import custom modules

stormtrack = 1
if stormtrack:
    
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
    
else:
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
   # sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/predict_amv/")
    
from amv import proc,viz
import scm
#import amv_dataloader as adl
#%% Further Edits

# Data that has been processed by [preproc_CESM1_LENS]
outpath   = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/hfdamping_HTR/"

# Location ofsalinity data regridded to atmospheric grid using [prep_mld_PIC]
# Name structure: SSS_FULL_HTR_bilinear_num##.nc
ssspath   = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/SSS/"

# Name of the variables
vnames    = ("SSS","TS") # Variables
fluxnames = ("FSNS","FLNS","LHFLX","SHFLX") # Fluxes
dimnames  = ("lat","lon","time")

#%% Get SSS

# Get list of regridded SSS
nclist = glob.glob(ssspath+"SSS_FULL_HTR_bilinear_num*.nc")
nclist.sort()
nens   = len(nclist)


# Compute Ensemble Average for SSS, load SSS datasets
ds_all_sss  = []
for e in tqdm(range(nens)):
    
    ds = xr.open_dataset(nclist[e])
    ds_all_sss.append(ds)
    if e == 0:
        ds          = ds.sel(time=slice("1920-01-01","2005-12-31"))
        ens_avg_sss = ds.SSS.values
    else:
        ens_avg_sss = ens_avg_sss + ds.SSS.values
ens_avg_sss = ens_avg_sss / nens

# Save ensemble average
savename    = "%sCESM1_htr_SSS_ensAVG.nc" % outpath
coordsdict  = {'time':ds.time,
               'z_t' :ds.z_t,
               'lat' :ds.lat,
               'lon' :ds.lon}
sss_ensavg  = xr.DataArray(ens_avg_sss,coords=coordsdict,dims=coordsdict,name="SSS")
sss_ensavg.to_netcdf(savename,encoding={"SSS": {'zlib': True}})



#%%



# Land Ice Mask
#mask      = adl.load_limask()
# Essentially as input, we want:
    # <lat x lon x time>




#%% Rewrite this part as appropriate

# >> list of netcdfs to process for each variable

#%% Looping for each ensemble member

#%% Step 1 (Formatting for Intake)

#%% Step 2 (Preprocessing)

#%% Step 3 (ENSO Removal)

#%% Step 4 (HFF Calculation)

