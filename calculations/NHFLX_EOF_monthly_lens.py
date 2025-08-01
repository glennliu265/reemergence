#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

NHFLX_EOF_monthly
========================

Compute EOF Forcing (NAO, EAP) for Fprime for a CESM1 LENS simulation
Uses Fprime output computed by calc_Fprime_lens.py
Currently written to run on Astraeus.

2024.06.28: Update to support calc_Fprime from hfcalc/ and different formats

Inputs:
------------------------
    
    varname : dims                              - units                 - processing script
    Fprime  : (time, ens, lat, lon)             [W/m2]                  calc_Fprime_lens


Outputs: 
------------------------

    varname : dims                              - units 
    eofs    : (mode,mon,ens,lat,lon)            - [W/m2/stdevPC]
    pcs     : (mode,mon,ens,yr)
    varexp  : (mode,mon,ens)
    
Output File Name: (Outputs to same path as Fprime)
    "%sEOF_Monthly_NAO_EAP_Fprime_%s_%s_NAtl.nc" % (fpath,dampstr,rollstr)

What does this script do?
------------------------
(1) Load in Fprime  d
For each ensemble member....
    (2) Perform EOF Analysis (for each month) and get regression patterns
    (3) Flip/Correct Sign
(4) Save Output Files

Script History
------------------------
# On 2024.02.08
# Copy of NHFLX_EOF_monthly from stochmod/preprocessing
# Copied Fprime calculation step from preproc_sm_inputs_SSS

Created on Thu Feb  8 16:43:44 2024
@author: gliu

"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
from tqdm import tqdm
import copy
import glob
import matplotlib as mpl

import os

# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "Astraeus"

# First Load the Parameter File

sys.path.append("../")
# First Load the Parameter File
cwd = os.getcwd()
sys.path.append(cwd + "/..")

import reemergence_params as rparams

# Paths and Load Modules
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
procpath    = pathdict['procpath']


rawpath1 = pathdict['raw_path']
dpath    = input_path + "damping/"
mldpath  = input_path + "mld/"

#%% User Edits

concat_ens  = True   # Set to True to concatenate ensemble members, False to do memberwise calculations
dataset     = "era5" #"CESM1_HTR"#"cesm1le_htr_5degbilinear"#"cesm2_pic"##"CESM1_HTR"



fix_feb     = True # Keep this here. changes this to false if dataset= era5
if dataset == "CESM1_HTR":
    # Fprime calulation settings
    ncstr1     = "CESM1LE_%s_NAtl_19200101_20050101_bilinear.nc"
    dampstr    = "nomasklag1" # Damping String  (see "load damping of choice")
    rollstr    = "nroll0"
    fpath      = rawpath1 #input_path + "forcing/" #"/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"
    fnc        = "CESM1_HTR_FULL_Fprime_timeseries_%s_%s_NAtl.nc" % (dampstr,rollstr)
    
    regstr     = "NAtl"
    
    # Implement mask
    maskpath   = None
    masknc     = None

elif dataset == "cesm1le_htr_5degbilinear":
    
    regstr     = "Global"
    
    ncstr1     = None
    fpath      = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"
    fnc        = "cesm1le_htr_5degbilinear_Fprime_timeseries_cesm1le5degqnet_nroll0_%s.nc" % regstr
    dampstr    = "cesm1le5degqnet"
    rollstr    = "nroll0"
    fpath      = rawpath1
    
    maskpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/masks/"
    masknc     = "cesm1_htr_5degbilinear_icemask_05p_year1920to2005_enssum.nc"
     
elif dataset == "era5":
    
    rawpath1   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/reanalysis/proc/NATL_proc_obs/" 
    
    
    dampstr     = "THFLXpilotObs"
    rollstr     = "nroll0"
    fpath       = rawpath1
    ncstr1      = "ERA5_Fprime_THFLX_timeseries_THFLXpilotObs_nroll0_NAtl.nc"
    fnc         = ncstr1
    
    maskpath   = None#""
    masknc     = None#""
    fix_feb    = False
    
else:# dataset == "cesm2_pic":
    

    dampstr    = "CESM2PiCqnetDamp"
    rollstr    = "nroll0"
    fpath      = rawpath1
    ncstr1     = "%s%s_Fprime_timeseries_%s_%s_NAtl.nc" %  (fpath,dataset,dampstr,rollstr)
    fnc        = ncstr1

    maskpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/masks/"
    masknc     = "cesm2_pic_limask_0.3p_0.05p_0200to2000.nc"
    
    
print("Performing EOF Analysis on Fprime of the following file:\n\t%s" % fnc)

# EOF parameters
bboxeof    = [-80,20,0,65]
N_mode     = 200 # Maxmum mode will be adjusted to number of years...


#%% Some functions


def anomalize(ds,fix_feb=True):
    if fix_feb:
        ds = proc.fix_febstart(ds)
    ds = ds - ds.mean('ens')
    ds = proc.xrdeseason(ds,check_mon=False)
    return ds

# -----------------------------------------------------------------------------
#%% Part (1): Load Fprime computed by calc_Fprime_lens
# -----------------------------------------------------------------------------

daf      = xr.open_dataset(fpath + fnc).Fprime.load()

if np.any(daf.lon.data > 180):
    print("Flipping variable to -180...")
    daf = proc.lon360to180_xr(daf)
    

if 'ens' not in list(daf.dims):
    print("adding singleton ensemble dimension ")
    daf  = daf.expand_dims(dim={'ens':[1,]},axis=1)
    nens = 1
else:
    nens = len(daf.ens)

daf      = anomalize(daf,fix_feb=fix_feb)

# -----------------------------------------------------------------------------
#%% Part (2): Perform EOF Analysis on Fprime (copy from NHFLX_EOF_monthly)
# -----------------------------------------------------------------------------

# Apply Mask if option is set
if maskpath is not None and masknc is not None:
    ds_mask = xr.open_dataset(maskpath+masknc).mask
    print("Applying mask %s" % (maskpath+masknc))
    
    if np.any(ds_mask.lon.data > 180):
        print("Flipping mask to -180...")
        ds_mask = proc.lon360to180_xr(ds_mask)
        
    
    daf,ds_mask = proc.resize_ds([daf,ds_mask])
    daf = daf * ds_mask


flxa     = daf # [Time x Ens x Lat x Lon] # Anomalize variabless

# Apply area weight
wgt    = np.sqrt(np.cos(np.radians(daf.lat.values))) # [Lat]
flxwgt = flxa * wgt[None,None,:,None]

# Select Region
flxreg = proc.sel_region_xr(flxwgt,bboxeof)


flxout     = flxreg.values
ntime,nens,nlatr,nlonr = flxout.shape
if concat_ens:
    # IMPORTANT NOTE (implement fix later)
    # Variable must be stacked as [ens x time x otherdims]
    if flxout.shape[0] != nens:
        ens_reshape_flag = True
        print("Warning, since ensemble dimension is NOT first, temporarily permuting array to ens x time")
        flxout = flxout.transpose(1,0,2,3)
    else:
        ens_reshape_flag = False
    print("Stacking Dimensions")
    flxout = flxout.reshape(nens*ntime,1,nlatr,nlonr)
    ntime,nens,nlatr,nlonr = flxout.shape
npts       = nlatr*nlonr
nyr        = int(ntime/12)

# Repeat for full variable
flxout_full= flxa.values
_,_,nlat,nlon=flxout_full.shape
if ens_reshape_flag:
    print("Permuting full variable")
    print("\tOriginal Shape %s" % str(flxout_full.shape))
    flxout_full = flxout_full.transpose(1,0,2,3)
    print("\tNew Shape %s" % str(flxout_full.shape))
npts_full  = nlat*nlon
if concat_ens:
    flxout_full = flxout_full.reshape(ntime,1,nlat,nlon)
print("\tFinal Shape %s" % str(flxout_full.shape))

# Check to see if N_mode exceeds nyrs
if N_mode > nyr:
    print("Requested N_mode exists the maximum number of years, adjusting....")
    N_mode=nyr

# Preallocate for EOF Analysis
eofall    = np.zeros((N_mode,12,nens,nlat*nlon)) * np.nan
pcall     = np.zeros((N_mode,12,nens,nyr)) * np.nan
varexpall = np.zeros((N_mode,12,nens)) * np.nan
    
# Loop for ensemble memmber
for e in tqdm(range(nens)):
    
    # Remove NaN Points
    flxens            = flxout[:,e,:,:].reshape(ntime,npts) #  Time x Space
    okdata,knan,okpts = proc.find_nan(flxens,0)
    _,npts_valid = okdata.shape
    
    # Repeat for full data
    flxens_full       = flxout_full[:,e,:,:].reshape(ntime,npts_full)
    okdataf,knanf,okptsf = proc.find_nan(flxens_full,0)
    _,npts_validf = okdataf.shape
    
    # Reshape to [yr x mon x pts]
    okdatar  = okdata.reshape(nyr,12,npts_valid)
    okdatarf = okdataf.reshape(nyr,12,npts_validf)
    
    # Calculate EOF by month
    for im in range(12):
        
        # Compute EOF
        datain          = okdatar[:,im,:].T # --> [space x time]
        eofs,pcs,varexp = proc.eof_simple(datain,N_mode,1)
        
        # Standardize PCs
        pcstd = pcs / pcs.std(0)[None,:]
        
        # Regress back to dataset
        datainf = okdatarf[:,im,:].T
        eof,b = proc.regress_2d(pcstd.T,datainf.T) # [time x pts]
        
        
        # Save the data
        eofall[:,im,e,okptsf] = eof.copy()
        pcall[:,im,e,:] = pcs.T.copy()
        varexpall[:,im,e] = varexp.copy()

# Reshape the variable
eofall = eofall.reshape(N_mode,12,nens,nlat,nlon) # (86, 12, 42, 96, 89)

# ----------------------------------------------------------------------------
#%% Part (3) Flip sign to match NAO+ (negative heat flux out of ocean/ -SLP over SPG)
# ----------------------------------------------------------------------------

spgbox     = [-60,20,40,80]
eapbox     = [-60,20,40,60] # Shift Box west for EAP

N_modeplot = 5

for N in tqdm(range(N_modeplot)):
    if N == 1:
        chkbox = eapbox # Shift coordinates west
    else:
        chkbox = spgbox
    for e in range(nens):
        for m in range(12):
            
            
            sumflx = proc.sel_region(eofall[N,[m],e,:,:].transpose(2,1,0),flxa.lon.values,flxa.lat.values,chkbox,reg_avg=True)
            #sumslp = proc.sel_region(eofslp[:,:,[m],N],lon,lat,chkbox,reg_avg=True)
            
            if sumflx > 0:
                print("Flipping sign for NHFLX, mode %i month %i" % (N+1,m+1))
                eofall[N,m,e,:,:]*=-1
                pcall[N,m,e,:] *= -1

# ----------------------------------------------------------------------------
#%% Part (4) Convert EOF to Data Array and save
# ----------------------------------------------------------------------------


startyr   = daf.time.data[0]
nyrs      = int(len(daf.time)/12)
if concat_ens:
    tnew      = np.arange(0,int(ntime/12))
else:
    tnew      = xr.cftime_range(start=startyr,periods=nyrs,freq="YS",calendar="noleap")

# Make Dictionaries
coordseof = dict(mode=np.arange(1,N_mode+1),mon=np.arange(1,13,1),ens=np.arange(1,nens+1,1),lat=flxa.lat,lon=flxa.lon)
daeof     = xr.DataArray(eofall,coords=coordseof,dims=coordseof,name="eofs")

coordspc  = dict(mode=np.arange(1,N_mode+1),mon=np.arange(1,13,1),ens=np.arange(1,nens+1,1),yr=tnew)
dapcs     = xr.DataArray(pcall,coords=coordspc,dims=coordspc,name="pcs")

coordsvar = dict(mode=np.arange(1,N_mode+1),mon=np.arange(1,13,1),ens=np.arange(1,nens+1,1))
davarexp  = xr.DataArray(varexpall,coords=coordsvar,dims=coordsvar,name="varexp")


ds_eof    = xr.merge([daeof,dapcs,davarexp])
edict_eof = proc.make_encoding_dict(ds_eof)

# ex. cesm2_pic_EOF_Monthly_NAO_EAP_Fprime_CESM2PiCqnetDamp_nroll0_NAtl.nc
savename  = "%s%s_EOF_Monthly_NAO_EAP_Fprime_%s_%s_%s.nc" % (fpath,dataset,dampstr,rollstr,regstr)
if concat_ens:
    savename = proc.addstrtoext(savename,"_concatEns",adjust=-1)

ds_eof.to_netcdf(savename,encoding=edict_eof)
print("Save output to %s" % savename)

#%%



# #%%
# #%% Visualize to check

# bboxchk = [-40,10,40,65]
# e       = 1
# N       = 0
# im      = 11

# fig,ax,mdict = viz.init_orthomap(1,1,bboxeof)
# ax = viz.add_coast_grid(ax,bbox=bboxeof)

# pcm = ax.pcolormesh(flxa.lon,flxa.lat,eofall[N,im,e,:,:],transform=mdict['noProj'])
# fig.colorbar(pcm,ax=ax)
# ax.set_title("EOF %02i Month %02i Ens %02i" % (N+1,im+1,e+1))

# ax = viz.plot_box(bboxchk,ax=ax)
#  # [Time x Ens x Lat x Lon]



