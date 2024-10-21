#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 09:01:03 2024

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

#%% 

#compare_name = "CESM1LE"

# Indicate files containing ACFs
cesm_name   = "CESM1_1920to2005_%sACF_lag00to60_ALL_ensALL.nc"
vnames      =  ["SST","SSS"] #["SST","SSS","TEMP"]
sst_expname = "SM_SST_Draft01_Rerun_QekCorr_SST_autocorrelation_thresALL_lag00to60.nc"#"SM_SST_EOF_LbddCorr_Rerun_SST_autocorrelation_thresALL_lag00to60.nc"
sss_expname = "SM_SSS_Draft01_Rerun_QekCorr_SSS_autocorrelation_thresALL_lag00to60.nc"#"SM_SSS_EOF_LbddCorr_Rerun_lbdE_neg_SSS_autocorrelation_thresALL_lag00to60.nc"

#sst_expname = "SM_SST_EOF_LbddCorr_Rerun_SST_autocorrelation_thresALL_lag00to60.nc"
#sss_expname = "SM_SSS_EOF_LbddCorr_Rerun_lbdE_neg_SSS_autocorrelation_thresALL_lag00to60.nc"

# Load Region Information
regionset       = "SSSCSU"
regiondicts     = rparams.region_sets[regionset]
bboxes          = regiondicts['bboxes']
regions_long    = regiondicts['regions_long']
rcols           = regiondicts['rcols']
rsty            = regiondicts['rsty']
regplot         = [0,1,3]
nregs           = len(regplot)

# Load Point Information
pointset        = "PaperDraft02"
ptdict          = rparams.point_sets[pointset]
ptcoords        = ptdict['bboxes']
ptnames         = ptdict['regions']
ptnames_long    = ptdict['regions_long']
ptcols          = ptdict['rcols']
ptsty           = ptdict['rsty']

#%%

ds_sst = xr.open_dataset(rawpath + "CESM1LE_Qek_SST_NAtl_19200101_20050101_bilinear.nc").load()
ds_sss = xr.open_dataset(rawpath + "CESM1LE_Qek_SSS_NAtl_19200101_20050101_bilinear.nc").load()

#%%

ds_all  = [ds_sst.Qek,ds_sss.Qek]
t       = 0
e       = 0
dsin    = [ds.isel(ensemble=e,time=t) for ds in ds_all]


diff = dsin[0]-dsin[1]

#%%


