#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Do a Linear Regression, but with mask array

Created on Mon Feb 12 16:39:34 2024

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

import numpy.ma as ma

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
    
# # Loop for ensemble memmber
# for e in tqdm(range(nens)):

#%% Old MEthod
e = 0
im = 0

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


# Compute EOF
datain          = okdatar[:,im,:].T # --> [space x time]
eofs,pcs,varexp = proc.eof_simple(datain,N_mode,1)

# Standardize PCs
pcstd = pcs / pcs.std(0)[None,:]

# Regress back to dataset
datainf = okdatarf[:,im,:].T
eof,b = proc.regress_2d(pcstd.T,datainf.T) # [time x pts]

    

#%% New method with masked array

# Make a numpy masked array
flxout_masked    = ma.array(flxout,mask=np.isnan(flxout))


# Reshape to time x pts
flxens           = flxout[:,e,:,:].reshape(ntime,npts)
flxens_yrmon     = flxens.reshape(nyr,12,npts)

# Compute EOF
datain           = flxens_yrmon[:,im,:].T
datainm          = ma.array(datain,mask=np.isnan(datain))
datainm2          = ma.compress_rowcols(datainm,axis=0) # Axis is axis to reduce points over. use compress_cols if it is [ntime x npts]
eofs,pcs,varexp  = proc.eof_simple(datainm2,N_mode,1)

# Replace into array
mask         = ma.getmask(datainm)
eofout       = np.zeros(datain.shape) * np.nan

#eofout       = np.where(~np.isnan(mask),)



eofout[~mask] = eofs.flatten() 
eofout = eofout.reshape(mask.shape)
eofout = eofout.reshape(nlat,nlon)

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
