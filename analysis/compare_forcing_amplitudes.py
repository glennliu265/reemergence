#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Load in forcing amplitudes with the missing terms
Written for SSS Paper Revisions

Copied upper section of viz_inputs_paper_draft.py

Created on Fri Apr 25 09:11:45 2025

@author: gliu

"""



import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import os
import matplotlib.patheffects as PathEffects
from cmcrameri import cm

# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "Astraeus"

# First Load the Parameter File
cwd = os.getcwd()
sys.path.append(cwd+ "/..")
sys.path.append("../")
import reemergence_params as rparams

# Paths and Load Modules
machine    = "Astraeus"
pathdict   = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])
from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx

# Set needed paths
figpath     = pathdict['figpath']
proc.makedir(figpath)
input_path  = pathdict['input_path']
output_path = pathdict['output_path']
rawpath     = pathdict['raw_path']

# ----------------------------------
#%% User Edits
# ----------------------------------

# Indicate the experiment
expname_sss         = "SSS_Revision_Qek_TauReg"#"#"SSS_Draft03_Rerun_QekCorr"#SSS_EOF_LbddCorr_Rerun_lbdE_neg" #"SSS_EOF_Qek_LbddEnsMean"#"SSS_EOF_Qek_LbddEnsMean"
expname_sst         = "SST_Revision_Qek_TauReg"#"SST_Draft03_Rerun_QekCorr"#"SST_EOF_LbddCorr_Rerun"


# Constants
dt          = 3600*24*30 # Timestep [s]
cp          = 3850       # 
rho         = 1026    #`23      # Density [kg/m3]
B           = 0.2        # Bowen Ratio, from Frankignoul et al 1998
L           = 2.5e6      # Specific Heat of Evaporation [J/kg], from SSS model document

fsz_tick    = 18
fsz_title   = 24
fsz_axis    = 22


debug       = False



#%% Add some functions to load (and convert) inputs

def stdsqsum(invar,dim):
    return np.sqrt(np.nansum(invar**2,dim))

def stdsq(invar):
    return np.sqrt(invar**2)

def stdsqsum_da(invar,dim):
    return np.sqrt((invar**2).sum(dim))

def convert_ds(invar,lat,lon,):
    
    if len(invar.shape) == 4: # Include mode
        nmode = invar.shape[0]
        coords = dict(mode=np.arange(1,nmode+1),mon=np.arange(1,13,1),lat=lat,lon=lon)
    else:
        coords = dict(mon=np.arange(1,13,1),lat=lat,lon=lon)
    
    return xr.DataArray(invar,coords=coords,dims=coords)

def compute_detrain_time(kprev_pt):
    
    detrain_mon   = np.arange(1,13,1)
    delta_mon     = detrain_mon - kprev_pt#detrain_mon - kprev_pt
    delta_mon_rev = (12 + detrain_mon) - kprev_pt # Reverse case 
    delta_mon_out = xr.where(delta_mon < 0,delta_mon_rev,delta_mon) # Replace Negatives with 12+detrain_mon
    delta_mon_out = xr.where(delta_mon_out == 0,12.,delta_mon_out) # Replace deepest month with 12
    delta_mon_out = xr.where(kprev_pt == 0.,np.nan,delta_mon_out)
    
    return delta_mon_out


#%% Plotting Params

mpl.rcParams['font.family'] = 'Avenir'
bboxplot                    = [-80,0,20,65]
proj                        = ccrs.PlateCarree()
#lon                         = daspecsum.lon.values
#lat                         = daspecsum.lat.values
mons3                       = proc.get_monstr()


plotver = "rev1" # [sub1]

#%% Load BSF and Ice Mask (copied from compare_detrainment_damping)

bsf      = dl.load_bsf()

# Load Land Ice Mask
icemask  = xr.open_dataset(input_path + "masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")

# Resize
bsf,icemask    = proc.resize_ds([bsf,icemask])
bsf_savg = proc.calc_savg_mon(bsf)

#
mask = icemask.MASK.squeeze()
mask_plot = xr.where(np.isnan(mask),0,mask)#mask.copy()

mask_apply = icemask.MASK.squeeze().values
#mask_plot[np.isnan(mask)] = 0


# Load Gulf Stream
ds_gs   = dl.load_gs()
ds_gs   = ds_gs.sel(lon=slice(-90,-50))
ds_gs2  = dl.load_gs(load_u2=True)

# Load velocities
ds_uvel,ds_vvel = dl.load_current()
tlon  = ds_uvel.TLONG.mean('ens').data
tlat  = ds_uvel.TLAT.mean('ens').data

# Load Region Information
regionset       = "SSSCSU"
regiondicts     = rparams.region_sets[regionset]
bboxes          = regiondicts['bboxes']
regions_long    = regiondicts['regions_long']
rcols           = regiondicts['rcols']
rsty            = regiondicts['rsty']
regplot         = [0,1,3]
nregs           = len(regplot)


# Get Point Info
pointset        = "PaperDraft02"
ptdict          = rparams.point_sets[pointset]
ptcoords        = ptdict['bboxes']
ptnames         = ptdict['regions']
ptnames_long    = ptdict['regions_long']
ptcols          = ptdict['rcols']
ptsty           = ptdict['rsty']


#%% Check and Load Params (copied from run_SSS_basinwide.py on 2024-03-04)

# Load the parameter dictionary
expparams_byvar     = []
paramset_byvar      = []
convdict_byvar      = []
convda_byvar        = []
for expname in [expname_sst,expname_sss]:
    
    print("Loading inputs for %s" % expname)
    
    expparams_raw   = np.load("%s%s/Input/expparams.npz" % (output_path,expname),allow_pickle=True)
    
    expparams       = scm.repair_expparams(expparams_raw)
    
    # Get the Variables (I think only one is really necessary)
    #expparams_byvar.append(expparams.copy())
    
    # Load Parameters
    paramset = scm.load_params(expparams,input_path)
    inputs,inputs_ds,inputs_type,params_vv = paramset
    

    # Convert to the same units
    convdict                               = scm.convert_inputs(expparams,inputs,return_sep=True)
    
    # Get Lat/Lon
    ds = inputs_ds['h']
    lat = ds.lat.data
    lon = ds.lon.data
    
    # Convert t22o DataArray
    varkeys = list(convdict.keys())
    nk = len(varkeys)
    conv_da = {}
    for nn in range(nk):
        #print(nn)
        varkey = varkeys[nn]
        invar  = convdict[varkey]
        conv_da[varkey] =convert_ds(invar,lat,lon)
        
    
    # Append Output
    expparams_byvar.append(expparams)
    paramset_byvar.append(paramset)
    convdict_byvar.append(convdict)
    convda_byvar.append(conv_da)

# --------------------------------------
#%% Load kprev and compute convert lbd-d
# --------------------------------------

lbdd_sst    = paramset_byvar[0][1]['lbd_d']
lbdd_sss    = paramset_byvar[1][1]['lbd_d']

# Compute Detrainment Times
ds_kprev    = xr.open_dataset(input_path + "mld/CESM1_HTR_FULL_kprev_NAtl_EnsAvg.nc")
delta_mon   = xr.apply_ufunc(
        compute_detrain_time,
        ds_kprev.kprev,
        input_core_dims=[['mon']],
        output_core_dims=[['mon']],
        vectorize=True,
        )

lbdd_sst_conv = -delta_mon / np.log(lbdd_sst)
lbdd_sss_conv = -delta_mon / np.log(lbdd_sss)

#%% Load (or compute) the SST Evaporation Feedback

# Load lbd_e
lbd_e    = xr.open_dataset(input_path + "forcing/" + expparams_byvar[1]['lbd_e']).lbd_e.load() # [mon x lat x lon]
lbd_e    = proc.sel_region_xr(lbd_e,bbox=expparams_byvar[1]['bbox_sim'])

# Convert [sec --> mon]
lbd_emon = lbd_e * dt
#lbd_emon = lbd_emon.transpose('lon','lat','mon')#.values
