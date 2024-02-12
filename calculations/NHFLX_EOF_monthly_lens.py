#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

NHFLX_EOF_monthly
========================

Compute EOF Forcing (NAO, EAP) for Fprime for a CESM1 LENS simulation
Uses Fprime output computed by calc_Fprime_lens.py
Currently written to run on Astraeus.

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
(1) Load in Fprime 
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

#%% Import Custom Modules
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx

#%% Locate Target File

stormtrack = 0

# Path to variables processed by prep_data_byvariable_monthly, Output will be saved to rawpath1
if stormtrack:
    rawpath1 = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/NATL_proc/"
    dpath    = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/damping/"
    mldpath  = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/mld/"
else:
    rawpath1 = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"
    mldpath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/mld/"
    dpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/damping/"
# /Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1

ncstr1   = "CESM1LE_%s_NAtl_19200101_20050101_bilinear.nc"

#%% User Edits

# Fprime calulation settings
dampstr    = "nomasklag1" # Damping String  (see "load damping of choice")
rollstr    = "nroll0"
fpath      = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"
fnc        = "%sCESM1_HTR_FULL_Fprime_timeseries_%s_%s_NAtl.nc" % (fpath,dampstr,rollstr)

# EOF parameters
bboxeof    = [-80,20,0,65]
N_mode     = 200 # Maxmum mode will be adjusted to number of years...

# -----------------------------------------------------------------------------
#%% Part (1): Load Fprime computed by calc_Fprime_lens
# -----------------------------------------------------------------------------

daf = xr.open_dataset(fnc).Fprime

# -----------------------------------------------------------------------------
#%% Part (2): Perform EOF Analysis on Fprime (copy from NHFLX_EOF_monthly)
# -----------------------------------------------------------------------------
flxa     = daf # [Time x Ens x Lat x Lon] # Anomalize variabless

# Apply area weight
wgt    = np.sqrt(np.cos(np.radians(daf.lat.values))) # [Lat]
flxwgt = flxa * wgt[None,None,:,None]

# Select Region
flxreg = proc.sel_region_xr(flxwgt,bboxeof)


flxout     = flxreg.values
ntime,nens,nlatr,nlonr = flxout.shape
npts       = nlatr*nlonr
nyr        = int(ntime/12)

# Repeat for full variable
flxout_full= flxa.values
_,_,nlat,nlon=flxout_full.shape
npts_full  = nlat*nlon

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
# Make Dictionaries
coordseof = dict(mode=np.arange(1,N_mode+1),mon=np.arange(1,13,1),ens=np.arange(1,43,1),lat=flxa.lat,lon=flxa.lon)
daeof     = xr.DataArray(eofall,coords=coordseof,dims=coordseof,name="eofs")

coordspc  = dict(mode=np.arange(1,N_mode+1),mon=np.arange(1,13,1),ens=np.arange(1,43,1),yr=np.arange(1920,2005+1))
dapcs     = xr.DataArray(pcall,coords=coordspc,dims=coordspc,name="pcs")

coordsvar = dict(mode=np.arange(1,N_mode+1),mon=np.arange(1,13,1),ens=np.arange(1,43,1))
davarexp  = xr.DataArray(varexpall,coords=coordsvar,dims=coordsvar,name="varexp")


ds_eof    = xr.merge([daeof,dapcs,davarexp])
edict_eof = proc.make_encoding_dict(ds_eof)

savename  = "%sEOF_Monthly_NAO_EAP_Fprime_%s_%s_NAtl.nc" % (fpath,dampstr,rollstr)

ds_eof.to_netcdf(savename,encoding=edict_eof)


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



