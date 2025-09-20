#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

calculate_Fprime_lens.py
========================

Given Damping, MLD, SST , and Qnet, compute Fprime where:
    
    Qnet = F' + lbd*T
So:
    F' = Qnet - lbd*T 
    
Does the same for E', where Qnet is replaced by QLHFLX
    E' = Qlhflx - lbd*T'


IMPT: This formulation assumes that Qnet is POSITIVE UPWARDS! (and that the)
    heat flux was computed with the same setting (i.e. +lambda with +T 
                                                 = increased upward heat flux)
    This script performs a check over the Gulf Stream wintertime and flips
    the sign of the heat flux if this is negative.
    
Where T and Qnet are **not anomalized**. Written to run on Astraeus...
This script will be used by NHFLX_EOF_monthly.


Inputs:
------------------------

    varname : dims                              - units                 - processing script
    SST     : (ensemble, time, lat, lon)        [degC]                  ????
    qnet    : (ensemble, time, lat, lon)        [W/m2]                  ????
    h       : (mon, ens, lat, lon)              [meters]                ????
    damping : (mon, ens, lat, lon)              [degC/W/m2] OR [1/mon]  ????

Outputs: 
------------------------

    varname : dims                              - units 
    Fprime  : (time, ens, lat, lon)             [W/m2]

Output File Name: "%sCESM1_HTR_FULL_Fprime_timeseries_%s_%s_NAtl.nc" % (rawpath1,dampstr,rollstr)

What does this script do?
------------------------
    1) Load, deseasonalize, and detrend qnet/TS
    2) Load (and optionally convert) HFF
    3) Tile HFF and compute Fprime
    4) Save output

Script History
------------------------
 - Moved from NHFLX_EOF_monthly_lens.py on 2024.02.12
 - Copied Fprime calculation step from preproc_sm_inputs_SSS
 - Created on Mon Feb 12 15:40:17 2024

@author: gliu
"""


import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt

#%% Import Custom Modules

# stormtrack




#%%



#%% Set Paths

#Eprime     = False # Set to True to Compute E' instead of F'

stormtrack   = 0

# Path to variables processed by prep_data_byvariable_monthly, Output will be saved to rawpath1
if stormtrack:
    rawpath1 = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/NATL_proc/"
    dpath    = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/damping/"
    mldpath  = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/mld/"
    
    amvpath = "/home/glliu/00_Scripts/01_Projects/00_Commons/" # amv module
    scmpath = "/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/" # scm module
    
else:
    rawpath1 = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"
    mldpath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/mld/"
    dpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/damping/"
    
    amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
    scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"
    
sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
    
#%% Additional functions

# taken from preprocess_ds in calc_hff_general_new:
    
def detrend_ds(dsa,detrend,lensflag,vname,ds_gmsst=None):
    
    # Detrend ----------------
    if lensflag:
        dsadt = dsa - dsa.mean('ens')
    elif detrend == "linear" or detrend == "1": # (1): Simple Linear Detrend (9.68s)
        dsadt    = proc.xrdetrend(dsa)
    elif detrend == "linearmon":
        dsadt    = proc.xrdetrend_nd(dsa,1,return_fit=False,regress_monthly=True)
    elif detrend == 'quadratic':
        dsadt    = proc.xrdetrend_nd(dsa,2,return_fit=False)
    elif detrend == "quadraticmon":
        dsadt    = proc.xrdetrend_nd(dsa,2,return_fit=False,regress_monthly=True)
    elif detrend == "GMSST":
        # (3): Removing GMSST
        gmout       = proc.detrend_by_regression(dsa,ds_gmsst.GMSST_MeanIce)
        dsadt        = gmout[vname]
    elif detrend == "GMSSTmon":
        gmoutmon    = proc.detrend_by_regression(dsa,ds_gmsst.GMSST_MeanIce,regress_monthly=True)
        dsadt        = gmoutmon[vname]
    else:
        print("No detrending will be performed...")
    
    return dsadt
#%% Load GMSST for detrending (note that this must be manually entered...)

# Load GMSST
dpath_gmsst = "/Users/gliu/Downloads/02_Research/01_Projects/05_SMIO/01_Data/"
nc_gmsst    = "ERA5_GMSST_1979_2024.nc"
ds_gmsst    = xr.open_dataset(
    dpath_gmsst + nc_gmsst).load()  # .GMSST_MeanIce.load()

#%%

# Calculation Options
calc_name = "era5" #"CESM1"

# Damping Options ----------
dampstr = "QNETgmsstMON" # Damping String  (see below, "load damping of choice")
Eprime  = False # Set to True to look for LHFLX instead of qnet (for CESm1 calculations)

if calc_name == "CESM1":
    # (Note this is the original calculation using CESM1)
    # Indicate Search String for qnet/SST files ------d
    ncstr1   = "CESM1LE_%s_NAtl_19200101_20050101_bilinear.nc"
    # Indicate Mixed-Layer Depth File
    mldnc    = "%sCESM1_HTR_FULL_HMXL_NAtl.nc" % mldpath
    
elif calc_name == "era5": 
    
    # sst,thflx
    rawpath1 = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/reanalysis/proc/NATL_proc_obs/"
    
    if dampstr == "THFLXpilotObs":
        ncstr1   = "ERA5_%s_NAtl_1979to2021.nc" # from hfcalc/scrap/process_crop_data.py
    else:
        ncstr1   = "ERA5_%s_NAtl_1979to2024.nc" # from hfcalc/scrap/process_crop_data.py
    mldnc    = mldpath + "MIMOC_regridERA5_h_pilot.nc" # Processed 
    

# Fprime Calculation Options
nroll    = 0
rollstr  = "nroll%0i"  % nroll



"""
Current List of Damping Strings
-- Name             -- ncfile                                               -- Description
"nomasklag1"        "CESM1_HTR_FULL_qnet_damping_nomasklag1.nc"             Default Qnet Damping as calculated from covariance-based method.
"Expfitlbda123"     "CESM1_HTR_FULL_Expfit_lbda_damping_lagsfit123.nc"      Exp Fit to SST - Expfit to SSS; Mean of Lags 1,2,3
"ExpfitSST123"      "CESM1_HTR_FULL_Expfit_SST_damping_lagsfit123.nc"       Exp Fit to SST (total); Mean of Lags 1,2,3
"LHFLXnomasklag1"   "CESM1_HTR_FULL_LHFLX_damping_nomasklag1_EnsAvg.nc"     Default LHFLX Damping as calculated from covariance-based method
"THFLXpilotObs"     "ERA5_thflx_damping_pilot.nc"                           THFLX Estimates for pilot run of observational stochastic model
"QNETpilotObs"      "ERA5_qnet_damping_pilot.nc"                            Qnet Estimates, 1979 to 2024 ERA5
"QNETpilotObsAConly" "ERA5_qnet_damping_AConly.nc"
"QNETgmsstMON"      "ERA5_qnet_damping_AConly_detrendGMSSTmon.nc"
"""

detrend = "1" # For original cases (non CESM, detrend using linear). Specify below for future cases
if dampstr == "Expfitlbda123":
    convert_wm2=True
    hff_nc   = "CESM1_HTR_FULL_Expfit_lbda_damping_lagsfit123.nc"
elif dampstr == "nomasklag1":
    convert_wm2=False
    hff_nc = "CESM1_HTR_FULL_qnet_damping_nomasklag1.nc"
elif dampstr == "ExpfitSST123":
    convert_wm2=True
    hff_nc   = "CESM1_HTR_FULL_Expfit_SST_damping_lagsfit123.nc"#"CESM1_HTR_FULL_qnet_damping_nomasklag1.nc"
elif dampstr == "LHFLXnomasklag1":
    convert_wm2= False
    hff_nc   = "CESM1_HTR_FULL_LHFLX_damping_nomasklag1.nc"
elif dampstr == "THFLXpilotObs":
    convert_wm2 = False
    hff_nc   = "ERA5_thflx_damping_pilot.nc"
    varnames = ['sst','thflx']
    flxname  = "thflx"
    vname_fn = "Fprime_THFLX"
elif dampstr == "QNETpilotObs":
    convert_wm2 = False
    hff_nc   = "ERA5_qnet_damping_pilot.nc"
    vname_fn = "Fprime_QNET"
    varnames = ['sst','qnet']
    flxname  = "qnet"
elif dampstr == "QNETpilotObsAConly":
    convert_wm2 = False
    
    hff_nc = "ERA5_qnet_damping_AConly.nc"
    vname_fn = "Fprime_QNET"
    varnames = ['sst','qnet']
    flxname  = "qnet"
elif dampstr == "QNETgmsstMON":
    convert_wm2 = False
    hff_nc      = "ERA5_qnet_damping_AConly_detrendGMSSTmon.nc"
    varnames    = ["sst",'qnet']
    flxname     = 'qnet'
    detrend     = "GMSSTmon"
    vname_fn = "Fprime_QNET"
    
else:
    print("Invalid dampstr, currently not supported...")




# Conversion Factors
dt  = 3600*24*30
cp0 = 3996
rho = 1026

# -----------------------------------------------------------------------------
#%% Part 1: Load, Deseasonalize, Detrend qnet and SST
# -----------------------------------------------------------------------------
# Note this was copied from preproc_sm_inputs_SSS.py
st       = time.time()

# Load TS, flux and preprocess -------------------------
if calc_name == "CESM1":
    if Eprime:
        print("Loading LHFLX to compute E'")
        flxname = "LHFLX"
    else:
        print("Loading Q_net to compute F'")
        flxname = "qnet"
    varnames = ["SST",flxname]
#elif calc_name == "era5":
    


ds_load  = [xr.open_dataset(rawpath1+ ncstr1 % vn).load() for vn in varnames]

# Check if upwards positive (should be positive over gulf stream...
bbox_gs    = [-80,-60,20,40]
ds_load[1] = proc.check_flx(ds_load[1],flxname=flxname,bbox_gs=bbox_gs)

# <Delete this later, I already wrote a function>
# test_flx = ds_load[1][flxname]
# flx_gs   = proc.sel_region_xr(test_flx,bbox_gs)
# flx_savg = flx_gs.groupby('time.season').mean('time')
# flx_wint = flx_savg.sel(season='DJF')
# wintsum  = flx_wint.sum(['lat','lon']).data.item()
# if wintsum < 0:
#     print("Warning, wintertime avg values are NEGATIVE over the Gulf Stream.")
#     print("\tSign will be flipped to be Positive Upwards (into the atm)")
#ds_load[1] = ds_load[1] * -1
# < End >

# Anomalize
ds_anom  = [proc.xrdeseason(ds) for ds in ds_load]


# Detrend
if calc_name == "CESM1":
    
    # Detrend by Ensemble mean
    ds_dt    = [ds-ds.mean('ensemble') for ds in ds_anom] # [ens x time x lat x lon]
    
    # Transpose to [mon x ens x lat x lon]
    ds_dt    = [ds.transpose('time','ensemble','lat','lon') for ds in ds_dt]
    
    # Extract dataarrays
    ds_dt    = [ds_dt[ii][varnames[ii]] for ii in range(2)]
    
else:
    
    
    # Do Detrending
    ds_dt    = [detrend_ds(ds_anom[ii][varnames[ii]],detrend,False,varnames[ii],ds_gmsst=ds_gmsst) for ii in range(2)]

    ds_dt    = [ds.expand_dims('ensemble',axis=1) for ds in ds_dt]
    
# Note: Do unit conversions for ERA5
if (calc_name == "era5") and (dampstr == "THFLXpilotObs"):
    print("Converting from day --> month for ERA5 ")
    dtday    = 60*60*24
    ds_dt[1] = ds_dt[1] / dtday

# -----------------------------------------------------------------------------
#%% Part 2: Load and Convert Damping
# -----------------------------------------------------------------------------


# Load HFF
dshff    = xr.open_dataset(dpath + hff_nc) # [mon x ens x lat x lon]

if 'ensemble' not in dshff.coords:
    print("Adding ensemble dimension")
    dshff = dshff.expand_dims('ensemble',axis=1) 

# Load mixed layer depth for conversion
ds_mld   = xr.open_dataset(mldnc)

# Check sizes, make sure they are all the same...
# if dampstr is not None: # Not sure why, but it seems that the hff default is wrongly cropped
#     ds_list = ds_dt + [dshff,ds_mld]
#     ds_rsz  = proc.resize_ds(ds_list)
#     ds_dt = ds_rsz[:2]
#     dshff = ds_rsz[2]
#     ds_mld = ds_rsz[3]

# Convert HFF (1/mon to W/m2 per degC) if needed
if convert_wm2:
    dshff = dshff.damping * (rho*cp0*ds_mld.h) / dt  *-1 #need to do a check for - value!!
else:
    dshff= dshff.damping

# Load output to numpy
hff     = dshff.data
sst     = ds_dt[0].data#.SST.values
qnet    = ds_dt[1].data#[flxname].values

# -----------------------------------------------------------------------------
#%% Part 3: Tile heat flux feedback and make Fprime 
# -----------------------------------------------------------------------------
ntime,nens,nlat,nlon        = qnet.shape # Check sizes and get dimensions for tiling
ntimeh,nensh,nlath,nlonh    = hff.shape
nyrs                        = int(ntime/12)
hfftile                     = np.tile(hff.transpose(1,2,3,0),nyrs)
hfftile                     = hfftile.transpose(3,0,1,2)
# Check plt.pcolormesh(hfftile[0,0,:,:]-hfftile[12,0,:,:]),plt.colorbar(),plt.show()

#% Calculate F'
Fprime       = qnet - hfftile*np.roll(sst,nroll,axis=0) # Minus is the correct way to go
#Fprime_minus = qnet - hfftile*np.roll(sst,nroll)

# -----------------------------------------------------------------------------
#%% Part 4: Save Fprime output (full timeseries) (Optional)
# -----------------------------------------------------------------------------

if calc_name == "CESM1":
    if Eprime:
        outvar = "LHFLX"
    else:
        outvar = "Fprime"
    savename = "%sCESM1_HTR_FULL_%s_timeseries_%s_%s_NAtl.nc" % (rawpath1,"Eprime",dampstr,rollstr)
else:
    
    savename = "%sERA5_%s_timeseries_%s_%s_NAtl.nc" % (rawpath1,vname_fn,dampstr,rollstr)
    outvar   = "Fprime"
    
coords   = dict(time=ds_dt[0].time.values,ens=ds_dt[0].ensemble.values,lat=dshff.lat.values,lon=dshff.lon.values)
daf      = xr.DataArray(Fprime,coords=coords,dims=coords,name=outvar)

edict    = {outvar:{'zlib':True}}
daf.to_netcdf(savename,encoding=edict)
print("Script ran to completion in %.2fs" % (time.time()-st))

# -----------------------------------------------------------------------------
#%% Part 5. Check Whitening at a point
# -----------------------------------------------------------------------------

#% Here's a debug section to check the power spectra and whitening using Fprime
lonf  = -30
latf  = 53
nroll = 0

# Get SST and Qnet
dspt = [proc.selpt_ds(ds,lonf,latf,) for ds in ds_dt]
sstpt,qnetpt = dspt[0].values,dspt[1].values # [SST,HFLX]
hffpt        = proc.selpt_ds(dshff,lonf,latf)

# Tile and Compute
nyrs         = int(len(ds_dt[0].time)/12)
hffpttile    = np.array([np.tile(hffpt.values[:,e][:],nyrs).flatten() for e in range(nens)]).T


Fp        = qnetpt + hffpttile*np.roll(sstpt,nroll)
Fpminus   = qnetpt - hffpttile*np.roll(sstpt,nroll)

Fpminusr1 = qnetpt - hffpttile*np.roll(sstpt,-2,axis=0)

#%% Look at spectra

ints    = [sstpt,qnetpt,Fp,Fpminus,Fpminusr1]
labs    = ["SST","Qnet","Fprime Plus","Fprime Minus","Fprime Minus Roll 1"]
#output  = scm.quick_spectra(ints,)

nsmooth  = 25
pct      = 0.1
dtin     = 3600*24*30

specvars = []
nvars    = len(ints)
for vv in range(nvars):
    tsens    = ints[vv]
    tsens    = [tsens[:,e] for e in range(nens)]
    specout  = scm.quick_spectrum(tsens, nsmooth, pct, dt=dtin,return_dict=True,
                                 make_arr=True)
    specvars.append(specout)
    
#%% PLot the ensemble mean
dtplot  = dtin
fig,axs = plt.subplots(2,1,constrained_layout=True,figsize=(12,10))

for vv in range(nvars):
    
    if vv == 0:
        ax = axs[0]
    else:
        ax = axs[1]
    #ax = axs[vv]
    plotspec = specvars[vv]['specs'].mean(0)/dtplot
    plotfreq = specvars[vv]['freqs'].mean(0) * dtplot
    ax.plot(plotfreq,plotspec,label=labs[vv])
    
    ax.axhline(np.var(ints[vv])/(plotfreq[-1] - plotfreq[0]))
    ax.legend()
    
#%% Look at ACF

tsmetrics = []
nvars    = len(ints)
for vv in range(nvars):
    tsens    = ints[vv]
    tsens    = [tsens[:,e] for e in range(nens)]
    tsm = scm.compute_sm_metrics(tsens)
    tsmetrics.append(tsm)
    
#%% Look at ACF

fig,axs = plt.subplots(nvars,1,constrained_layout=True,figsize=(12,10))
lags = np.arange(37)
for vv in range(nvars):
    
    ax,_ = viz.init_acplot(7,lags,lags,ax=ax)
    ax = axs[vv]
    plotspec = np.array(tsmetrics[vv]['acfs'][7]).mean(0)
    plotfreq = np.arange(37)
    ax.plot(plotfreq,plotspec,label=labs[vv])
    
    #ax.axhline(np.var(ints[vv])/(plotfreq[-1] - plotfreq[0]))
    ax.legend()

#%% Check the pattern of Fprime

fstd = np.nanmax(np.abs(Fprime),0).squeeze()
plt.pcolormesh(fstd,vmin=400,vmax=500),plt.colorbar()
plt.pcolormesh(fstd,vmin=0,vmax=55),plt.colorbar()


#%% 
    