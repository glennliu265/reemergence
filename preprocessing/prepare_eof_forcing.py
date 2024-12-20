#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Combine regress_EOF_forcing and correct_eof_forcing_SSS

(1) Load NAO
(2) Load full variable
(3) Perform Regression to NAO
(4) Perform EOF Filtering
(5) Perform EOF variance correction
(6) Save the output



Inputs:
------------------------
    
    varname : dims                              - units                 - processing script
    <vname>   : (ens, time, lat, lon)           [varunits]              prep_data_byvariable_monthly
    
Outputs:
------------------------ 
    varname : dims                              - units                 - Notes
    <vname>   : (mode, ens, mon, lat, lon)           [varunits]              

What the script does
1) Detrend and Deseason variable
2) Load in EOF results and regress output there


Created on Thu Feb 15 13:09:47 2024

@author: gliu
"""

import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys
import cmocean
import xarray as xr
import time
from tqdm import tqdm


#%%


#




#%%





#%% Import modules

stormtrack      = 1

if stormtrack == 0:
    
    projpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/model_output/'
    outpathdat  = datpath + '/proc/'
    figpath     = projpath + "02_Figures/20240207"
    
    lipath      = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/landicemask_enssum.npy"
    rawpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"
    
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
    
    # Path of model input
    outpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/forcing/"

elif stormtrack == 1:
    
    datpath     = ""#"/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_output/"
    rawpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/forcing/"#/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/"
    outpathdat  = datpath + '/proc/'
    
    # Forcing Output Path
    fpath       = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/forcing/"
    
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")

from amv import proc,viz
import scm
import tbx

if stormtrack == 0: # Haven't added a figpath yet for stortmrack
    proc.makedir(figpath)

#%% User Edits

Eprime = True # Set to True to Load Eprime as LHFLX 


dataset_name = "cesm1le_htr_5degbilinear"


if dataset_name == "CESM1_HTR":
    

    # Path to variables processed by prep_data_byvariable_monthly
    rawpath1 = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/NATL_proc/"
    ncstr1   = "CESM1LE_%s_NAtl_19200101_20050101_bilinear.nc"

    # Path to variables processed by combine_precip
    rawpath2 = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/PRECIP/HTR_FULL/"
    ncstr2   = "%s_HTR_FULL.nc"
    
    # Path to Eprime processed by calc_Fprime_lens
    rawpath3 = rawpath1
    ncstr3   = "CESM1_HTR_FULL_Eprime_timeseries_LHFLXnomasklag1_nroll0_NAtl.nc"

    # EOF Information
    dampstr    = "nomasklag1"
    rollstr    = "nroll0"
    eofname    = "%sEOF_Monthly_NAO_EAP_Fprime_%s_%s_NAtl.nc" % (rawpath1,dampstr,rollstr)
    
    
    
elif dataset_name == "cesm1le_htr_5degbilinear":
    
    regstr      = "Global"
    dampstr     = "cesm1le5degLHFLX"
    rollstr     = "nroll0"
    
    
    # Path to variables processed by <hfcalc/Main/regrid_cesm1_lens.py>
    rawpath1    = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/output/proc/"
    ncstr1      = "cesm1_htr_5degbilinear_%s_" + regstr + "_1920to2005.nc"
    
    # Path to variables preprocessed by combine_precip (made the output the same...)
    rawpath2    = rawpath1
    ncstr2      = ncstr1
    
    # Path to Eprime processed by hfcalc/Main/calc_Fprime
    rawpath3    = rawpath1
    ncstr3      = "%s%s_Eprime_timeseries_%s_%s_%s.nc" % (rawpath1,dataset_name,dampstr,rollstr,regstr)
    
    # EOF Information
    dampstr_qnet    = "cesm1le5degqnet"  
    eofname         = "%s%s_EOF_Monthly_NAO_EAP_Fprime_%s_%s_%s.nc" % (rawpath1,dataset_name,dampstr_qnet,rollstr,regstr)
    
    
# Set Constants
omega = 7.2921e-5 # rad/sec
rho   = 1026      # kg/m3
cp0   = 3996      # [J/(kg*C)]
mons3 = proc.get_monstr()#('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')

#varname     = "SST"

# Variables to Load, example formats: 
#     - CESM1LE_LHFLX_NAtl_19200101_20050101_bilinear.nc
#     - 

varnames_in = ["LHFLX","PRECTOT"]
debug       = True  # Set to True to visualize for debugging



# Fprime or Qnet
correction     = True # Set to True to Use Fprime (T + lambda*T) instead of Qnet
correction_str = "_Fprime_rolln0" # Add this string for loading/saving

#%% Detrend and deseason select variable

# Load, detrend and deseason variables

# Load the files (Precipitation) ------------------------
ncprec  = ncstr2 % "PRECTOT" 
dsp     = xr.open_dataset(rawpath2+ncprec).load()

# Load the files, and generally all other files (LHFLX)
vn      = "LHFLX"
if Eprime:
    print("Loading Eprime as LHFLX")
    dslhf   = xr.open_dataset(ncstr3).load()
else:
    dslhf   = xr.open_dataset(rawpath1+ ncstr1 % vn).load()
if 'ensemble' in dslhf.dims:
    dslhf   = dslhf.rename({'ensemble':'ens'})

# Loop and preprocess
ds_in     = [dsp,dslhf]
ds_in     = [proc.fix_febstart(ds) for ds in ds_in]
ds_vnames = ["PRECTOT","Eprime"]

# Remove seasonal cycle
ds_anom  = [proc.xrdeseason(ds) for ds in ds_in]

# Detrend by removing ensemble average
ds_dt    = [ds - ds.mean('ens') for ds in ds_anom]

#%% Regress Selected Variable to NAO Principle Components

# Read in the NAO Data
dsnao = xr.open_dataset(eofname)
pcs   = dsnao.pcs.load() # [mode x mon x ens x yr]
nmode,nmon,nens,nyr  =pcs.shape

# Standardize PC
pcstd = pcs / pcs.std('yr')


#%% Perform Regressions

# Make into a simple function
def regress_pc_monens(ds,pcstd,vname,return_ds=True):
    
    # Given standardized PCs (mon, ens)
    # Regress the corresponding variable [vname] in ds (ens x yr x mon x lat x lon)
    # Get dimensions
    
    ds    = ds.transpose('ens','time','lat','lon')
    nens,ntime,nlat,nlon = ds.shape
    npts  = nlat*nlon
    nyr   = int(ntime/12)
    nmon  = 12
    
    # Combine spatial dimensions and loop
    invar        = ds.values
    invar        = invar.reshape(nens,nyr,nmon,npts)
    regr_pattern = np.zeros((nens,nlat*nlon,nmon,nmode)) * np.nan # [ens, space, month, mode]
    for e in tqdm(range(nens)):
        for im in range(nmon):
            # Select month and ensemble
            pc_mon    = pcstd.isel(mon=im,ens=e).values # [mode x year]
            invar_mon = invar[e,:,im,:] # [year x pts]
            
            # Get regression pattern
            rpattern,_=proc.regress_2d(pc_mon,invar_mon,verbose=False,nanwarn=True)
            # try:
            #     rpattern,_=proc.regress_2d(pc_mon,invar_mon,verbose=False)
            # except:
            #     print(pc_mon.shape)
            #     print(invar_mon.shape)
            #     rpattern,_=proc.regress_2d(pc_mon,invar_mon,verbose=False)
            regr_pattern[e,:,im,:] = rpattern.T.copy()
    regr_pattern = regr_pattern.reshape(nens,nlat,nlon,nmon,nmode)
    
    if return_ds:
        cout         = dict(ens=ds.ens,lat=ds.lat,lon=ds.lon,mon=np.arange(1,13,1),mode=pcstd.mode)
        da_out       = xr.DataArray(regr_pattern,coords=cout,dims=cout,name=vname)
        da_out       = da_out.transpose('mode','ens','mon','lat','lon')
        return da_out
    return regr_pattern


# Perform Regression in a loop
nvars = len(ds_in)
regression_maps = []
for v in range(nvars):
    
    # Get Variable and sizes
    vname  = ds_vnames[v] 
    ds     = ds_dt[v][vname]
    
    da_out = regress_pc_monens(ds,pcstd,vname)
    regression_maps.append(da_out)


#%% Save the Output

for v in range(nvars):
    
    # Save All Ensemble Members
    vname    = ds_vnames[v] 
    da_out   = regression_maps[v]
    edict    = {vname:{'zlib':True}}
    
    # Indicate the output variable name
    if Eprime and vname == "LHFLX": # Only change the name foor saving the file
        outvname = "Eprime"
    else:
        outvname = ds_vnames[v]
        
    # Set the save name
    if dataset_name == "CESM1_HTR":
        savename = "%sCESM1_HTR_FULL_%s_EOF_%s_%s_NAtl.nc" % (fpath,outvname,dampstr,rollstr) # ('mode','ens','mon','lat','lon')
    else:
        savename = "%s%s_%s_EOF_%s_%s_%s.nc" % (fpath,dataset_name,outvname,dampstr,rollstr,regstr)
    
    # Save the output (all ensemble members)
    da_out.to_netcdf(savename,encoding=edict)
    
    # Save Ens Avg
    savename_ea = proc.addstrtoext(savename,"_EnsAvg",adjust=-1)
    da_out_ensavg = da_out.mean('ens')
    da_out_ensavg.to_netcdf(savename_ea,encoding=edict)
    print("Saved %s to %s!" % (vname,savename_ea))
    
    




Created on Mon Aug 26 18:35:31 2024

@author: gliu

"""



