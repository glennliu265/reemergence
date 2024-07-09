#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Prepare Experiment Folder for Model Output or Observations
Similar to style of stochastic model output.

What this does:
    - Fix Febstart
    - Rotate Longitude to -180 to 180
    - Crop to stochastic model simulation region
    - Renames lat/lon/ens/time 
    - Creates Experiment Folder and Name

Works with scripts processed by:
    hfcalc/Main/

Created on Thu Jul  4 13:00:19 2024

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
import os

# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine    = "stormtrack"

# First Load the Parameter File
cwd = os.getcwd()
sys.path.append(cwd+"/../")
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
lipath      = pathdict['lipath']

# Make Needed Paths
proc.makedir(figpath)


#%%

# User Edits
dataset_name = "cesm2_pic"
varname      = "SST"
vname_in     = "TS"
bbox_cut     = [-80,0,20,65]

ncname       = "cesm2_pic_TS_NAtl_0200to2000.nc"
ncpath       = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/output/proc/"


expname       = "%s_%s" % (varname,dataset_name,)
print("Creating Experiment Directory for %s" % expname)

#%% Make the directory


expdir = output_path + expname + "/"
proc.makedir(expdir + "Input")
proc.makedir(expdir + "Output")
proc.makedir(expdir + "Metrics")
proc.makedir(expdir + "Figures")


outpath_data = expdir + "Output/"

#%% Actually, I will pause here because I didn't actually copy the files over yet?

runid        = 0

# Load the file
ds           = xr.open_dataset(ncpath + ncname)[vname_in].load()

# Fix February Start
ds           = proc.fix_febstart(ds)

# Rotate Longitude
ds           = proc.format_ds(ds)

# Crop region
dsreg        = proc.sel_region_xr(ds,bbox_cut)

# Rename Variable
dsreg       = dsreg.rename(varname)

# Save the output
edict       = proc.make_encoding_dict(dsreg)
outname     = "%s%s_runid%02i.nc" % (outpath_data,varname,runid)
print("Output will be resaved to %s" % outname)
dsreg.to_netcdf(outname,encoding=edict)


