#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Convert SM Output generated by original scripts from the stochmod repo
To the output that is processable in the reemergence module

Created on Thu Aug  1 15:25:31 2024

@author: gliu

"""

import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import scipy as sp

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

bboxplot                        = [-80,0,20,65]
mpl.rcParams['font.family']     = 'Avenir'
mons3                           = proc.get_monstr(nletters=3)

fsz_tick                        = 18
fsz_axis                        = 20
fsz_title                       = 16

rhocrit                         = proc.ttest_rho(0.05,2,86)

proj                            = ccrs.PlateCarree()


#%% 

# Check the runid readme for more infromation
fpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_output/"
fname = "stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0_1000yr_run2%02i_ampq0_method4_dmp0.npz"
runids = np.arange(10)

expname = "SST_cesm1pic_StochmodPaperRun"

#%% Load the Data

ldall = []
for rr in range(len(runids)):
    rid = runids[rr]
    ld = np.load(fpath+fname % rid,allow_pickle=True) # hconfig x lon x lat x time
    ldall.append(ld)
    

#%% Make the Experiment Directory

expdir = output_path + expname + "/"
proc.makedir(expdir + "Input")
proc.makedir(expdir + "Output")
proc.makedir(expdir + "Metrics")
proc.makedir(expdir + "Figures")

# Set additional output path
outpath_data = expdir + "Output/"


#%% Organize the files

# Make the time dimension
nlon,nlat,ntime = ssts[0].shape
times           = xr.cftime_range(start='0000',periods=ntime,freq="MS",calendar="noleap")

ssts = [ld['sst'][-1,:,:,:] for ld in ldall]

ssts = [sst.transpose(2,1,0) for sst in ssts]

#%%

# Make into xarray, and save for each one
coords  = dict(time=times,lat=ldall[0]['lat'],lon=ldall[0]['lon'])
da_ssts = [xr.DataArray(sst,coords=coords,dims=coords,name="SST") for sst in ssts]
edict   = proc.make_encoding_dict(da_ssts[0])


for rr in range(len(runids)):
    savename = "%sSST_runidrun2%02i.nc" % (outpath_data,runids[rr])
    print(savename)
    da_ssts[rr].to_netcdf(savename,encoding=edict)

#%%

