#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Copied sections of compare_regional_metrics, but streamlined to load for both SST/SSS for the paper

Copy upper section of viz_pointwise_variance


Created on Sat Aug 31 16:05:48 2024

@author: gliu
"""

from amv import proc, viz
import scm
import amv.xrfunc as xrf
import amv.loaders as dl
import reemergence_params as rparams
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
sys.path.append(cwd + "/..")

# Paths and Load Modules
pathdict = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])

# Set needed paths
figpath = pathdict['figpath']
input_path = pathdict['input_path']
output_path = pathdict['output_path']
procpath = pathdict['procpath']
rawpath = pathdict['raw_path']


# %% Import Custom Modules

# Import AMV Calculation

# Import stochastic model scripts

proc.makedir(figpath)

# %%

bboxplot = [-80, 0, 20, 65]
mpl.rcParams['font.family'] = 'Avenir'
mons3 = proc.get_monstr(nletters=3)

fsz_tick = 18
fsz_axis = 20
fsz_title = 16

rhocrit = proc.ttest_rho(0.05, 2, 86)

proj = ccrs.PlateCarree()


# %%  Indicate Experients (copying upper setion of viz_regional_spectra )
regionset       = "SSSCSU"
comparename     = "Paper_Draft02_AllExps"


# # #  Same as comparing lbd_e effect, but with Evaporation forcing corrections !!


# Take single variable inputs from compare_regional_metrics and combine them
# SSS Plotting Params
comparename_sss         = "SSS_Paper_Draft02"
expnames_sss            = ["SSS_Draft01_Rerun_QekCorr", "SSS_Draft01_Rerun_QekCorr_NoLbde",
                       "SSS_Draft01_Rerun_QekCorr_NoLbde_NoLbdd", "SSS_CESM"]
expnames_long_sss       = ["Stochastic Model ($\lambda^e$, $\lambda^d$)","Stochastic Model ($\lambda^d$)","Stochastic Model","CESM1"]
expnames_short_sss      = ["SM_lbde","SM_no_lbde","SM_no_lbdd","CESM"]
ecols_sss               = ["magenta","forestgreen","goldenrod","k"]
els_sss                 = ['dotted',"solid",'dashed','solid']
emarkers_sss            = ['+',"d","x","o"]


# # SST Comparison (Paper Draft, essentially Updated CSU) !!
# SST Plotting Params
comparename_sst     = "SST_Paper_Draft02"
expnames_sst        = ["SST_Draft01_Rerun_QekCorr","SST_Draft01_Rerun_QekCorr_NoLbdd","SST_CESM"]
expnames_long_sst   = ["Stochastic Model","Stochastic Model (No $\lambda^d$)","CESM1"]
expnames_short_sst  = ["SM","SM_NoLbdd","CESM"]
ecols_sst           = ["forestgreen","goldenrod","k"]
els_sst             = ["solid",'dashed','solid']
emarkers_sst        = ["d","x","o"]



expnames        = expnames_sst + expnames_sss
expnames_long   = expnames_long_sst + expnames_long_sss
ecols           = ecols_sst + ecols_sss
els             = els_sst + els_sss
emarkers        = emarkers_sst + emarkers_sss
expvars         = ["SST",] * len(expnames_sst) + ["SSS",] * len(expnames_sss)

#%%
##%% IMport other plotting stuff

# Load Current
ds_uvel,ds_vvel = dl.load_current()

# load SSS Re-emergence index (for background plot)
ds_rei = dl.load_rei("SSS_CESM",output_path).load().rei

# Load Gulf Stream
ds_gs = dl.load_gs()
ds_gs = ds_gs.sel(lon=slice(-90,-50))

# Load 5deg mask
maskpath = input_path + "masks/"
masknc5  = "cesm1_htr_5degbilinear_icemask_05p_year1920to2005_enssum.nc"
dsmask5 = xr.open_dataset(maskpath + masknc5)
dsmask5 = proc.lon360to180_xr(dsmask5).mask.drop_duplicates('lon')

masknc = "CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc"
dsmask = xr.open_dataset(maskpath + masknc).MASK.load()

maskin = dsmask


ds_gs2          = dl.load_gs(load_u2=True)

# Load Gulf Stream
ds_gs2  = dl.load_gs(load_u2=True)

# Load velocities
ds_uvel,ds_vvel = dl.load_current()
tlon  = ds_uvel.TLONG.mean('ens').data
tlat  = ds_uvel.TLAT.mean('ens').data

# Load Land Ice Mask
icemask     = xr.open_dataset(input_path + "masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")


mask        = icemask.MASK.squeeze()
mask_plot   = xr.where(np.isnan(mask),0,mask)#mask.copy()


mask_reg_sub    = proc.sel_region_xr(mask,bboxplot)
mask_reg_ori    = xr.ones_like(mask) * 0
mask_reg        = mask_reg_ori + mask_reg_sub


mask_apply  = icemask.MASK.squeeze().values

#%% Load the Pointwise Variance (Copied from compare_regional_metrics)

nexps       = len(expnames)

seavar_all  = []
var_all     = []
tsm_all     = []
rssts_all   = []
acfs_all    = []
amv_all     = []
for e in range(nexps):
    
    # Get Experiment information
    expname        = expnames[e]
    
    varname = expvars[e]
    # if "SSS" in expname:
    #     varname = "SSS"
    # elif "SST" in expname:
    #     varname = "SST"
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