#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

calc_ekman_advection_htr
========================

Description Here

Inputs:
------------------------
    
    varname : dims                              - units                 - processing script
    SSS/SST : (ensemble, time, lat, lon)        [psu]/[degC]            ???
    pcs     : (mode, mon, ens, yr)              [pc]                    NHFLX_NAO_monthly_lens


Outputs: 
------------------------

    varname : dims                              - units                 - Full Name
--- Mean SST/SSS Gradients ---
    dTdx    : (ensemble, month, lat, lon)       [degC/meter]            Forward Difference (x)
    dTdy    : (ensemble, month, lat, lon)       [degC/meter]            Forward Difference (y)
    dTdx2   : (ensemble, month, lat, lon)       [degC/meter]            Centered Difference (x)
    dTdy2   : (ensemble, month, lat, lon)       [degC/meter]            Centered Difference (y)
    
--- NAO-related Wind Stress ---
    TAUX    : (mode, ens, mon, lat, lon)
    TAUY    : (mode, ens, mon, lat, lon)

--- Ekman Forcing and Advection
    Qek     : (mode, mon, ens, lat, lon)        [W/m2/stdevEOF] or [psu/mon] Ekman Forcing
    Uek     : (mode, mon, ens, lat, lon)        [m/s/stdevEOF]          Eastward Ekman velocity
    Vek     : (mode, mon, ens, lat, lon)        [m/s/stdevEOF]          Northward Ekman velocity
     
--- 
    
Output File Name: 
    - Gradients (Forward) : CESM1_HTR_FULL_Monthly_gradT_<SSS>.nc
    - Gradients (Centered): CESM1_HTR_FULL_Monthly_gradT2_<SSS>.nc
    - NAO Wind Regression : CESM1_HTR_FULL_Monthly_TAU_NAO_%s_%s.nc % (rawpath,dampstr,rollstr)
    - Ekman Forcing and Advection: CESM1_HTR_FULL_Qek_%s_NAO_%s_%s_NAtl_EnsAvg.nc % (outpath,varname,dampstr,rollstr)
    

What does this script do?
------------------------
(1) Computes Mean Temperature/Salinity Gradients 
(2) Load in + anomalize wind stress (and obtain regressions to NAO, if option is set)
(3A) Method 1 (RegTau): Use EOF-regressed wind stress anomalies to get Uek and Qek 
(3B) Method 2 (DirReg): Compute full Qek term, then regress Qek' to NAO
(4) Perform EOF Filtering and Corrections

Script History
------------------------
 - [2025.04.03] Did some cleaning, rewrote NAO regression part to support concat_ENS
 - Copied calc_ekman_advection from stochmod on 2024.02.06
 - Calculate Ekman Advection for the corresponding EOFs and save
 - Uses variables processed by investigate_forcing.ipynb
 
Issues
------------------------
 - NAO Regression seems to yield opposite signs for wind stress, but added a correction for now

@author: gliu
"""

import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys
import cmocean
import xarray as xr
import time
from   tqdm import tqdm

#%% Import modules

stormtrack    = 1

if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/model_output/'
    outpathdat  = datpath + '/proc/'
    figpath     = projpath + "02_Figures/20240207"
   
    lipath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/landicemask_enssum.npy"
    rawpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"
    
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
    
    # Path of model input
    outpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/forcing/"

elif stormtrack == 1:
    #datpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_output/"
    #rawpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/"
    rawpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/NATL_proc/"
    datpath     = rawpath
    outpathdat  = datpath + '/proc/'
    
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
    figpath     = "/home/glliu/02_Figures/00_Scrap/"
    
    outpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/forcing/"

from amv import proc,viz
import scm
import tbx

proc.makedir(figpath)

# ==================================
#%% User Edits, Calculation Options
# ==================================

# Calculation Options ---------------------------------------
concat_ens      = True  # True to concatenate ensemble members along time axis
centered        = True  # Set to True to load centered-difference temperature
debug           = True  # Set to True to visualize for debugging
regress_nao     = True # Set to True to compute Qek based on wind stress regressed to NAO. Otherwise, use stdev(taux/tauy anoms)

# Set to False to skip a step... 
calc_dT         = False # Set to True to recalculate temperature gradients (Part 1)
calc_dtau       = False # Set to True to perform wind-stress regressions to PCs (Part 2)
calc_qek        = True  # set to True to calculate ekman forcing (Part 3)

# Select Fprime or Qnet for NAO Loading ---------------------------------------

correction      = True # Set to True to Use Fprime (T + lambda*T) instead of Qnet
correction_str  = "_Fprime_rolln0" # Add this string for loading/saving

convert_Wm2 = False # Set to True to convert SST Qek to Wm2
# Note: If true, multiplies output of NAO regression with -1
# so that circulation is cyclonic around icelandic low
# as of 2025.04.03, it seems that the values are flipped,
# possibly due to choice of (1) defining F' as positive downwards and (2) flipping
# the principle component for correction or (3) flipping the wind stress...
correct_TAU_sign = True # Multiple regression output by 1

# Option to Crop North Atlantic Region (for global 5deg Output) ---------------
crop_sm         = False
bbox_crop       = [-90,0,0,90]
regstr_crop     = "NAtl"

# Set Constants ---------------------------------------
omega           = 7.2921e-5 # rad/sec
rho             = 1026      # kg/m3
cp0             = 3996      # [J/(kg*C)]
mons3           = proc.get_monstr()

#%%
# CESM1 LE REGRID 5deg Inputs <Start> -----------------------------------------
regstr          = "Global"
varname         = "TS"
#rawpath         = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/output/proc/"
ncname_var      = "cesm1_htr_5degbilinear_%s_%s_1920to2005.nc" % (varname,regstr)
savename_grad   = "%scesm1_htr_5degbilinear_Monthly_gradT_%s_%s.nc" % (rawpath,varname,regstr)

# Wind Stress Information
tauxnc          = "cesm1_htr_5degbilinear_TAUX_%s_1920to2005.nc" % regstr#ncname_var % "TAUX"
tauync          = "cesm1_htr_5degbilinear_TAUY_%s_1920to2005.nc" % regstr #ncname_var % "TAUY"

# EOF Information
dampstr         = "cesm1le5degqnet"
rollstr         = "nroll0"
eofname         = "cesm1le_htr_5degbilinear_EOF_Monthly_NAO_EAP_Fprime_cesm1le5degqnet_nroll0_%s.nc" % regstr
savename_naotau = "%scesm1_htr_5degbilinear_Monthly_TAU_NAO_%s_%s_%s.nc" % (rawpath,dampstr,rollstr,regstr)

# MLD Info
mldpath         = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/mld/"
mldnc           = "cesm1_htr_5degbilinear_HMXL_%s_1920to2005.nc" % regstr

# Qek Information
nc_qek_out      =  "%scesm1_htr_5degbilinear_Qek_%s_NAO_%s_%s_%s.nc" % (outpath,varname,dampstr,rollstr,regstr)
savename_uek    =  "%scesm1_htr_5degbilinear_Uek_NAO_%s_%s_%s.nc" % (outpath,dampstr,rollstr,regstr)

# CESM1 LE REGRID 5deg <End> --------------------------------------------------

#%%
# CESM1 LE Inputs <Start> -----------------------------------------------------
varname         = "SST"
rawpath         = rawpath # Read from above
ncname_var      = "CESM1LE_%s_NAtl_19200101_20050101_bilinear.nc" % varname
savename_grad   = "%sCESM1_HTR_FULL_Monthly_gradT_%s.nc" % (rawpath,varname)

# Wind Stress Information
tauxnc          = "CESM1LE_TAUX_NAtl_19200101_20050101_bilinear.nc"
tauync          = "CESM1LE_TAUY_NAtl_19200101_20050101_bilinear.nc"

# EOF Information
dampstr         = "nomasklag1"
rollstr         = "nroll0"
eofname         = "%sCESM1_HTR_EOF_Monthly_NAO_EAP_Fprime_%s_%s_NAtl.nc" % (rawpath,dampstr,rollstr)
savename_naotau = "%sCESM1_HTR_FULL_Monthly_TAU_NAO_%s_%s.nc" % (rawpath,dampstr,rollstr)
if concat_ens:
    
    eofname         = proc.addstrtoext(eofname,"_concatEns",adjust=-1)
    savename_naotau = proc.addstrtoext(savename_naotau,"_concatEns",adjust=-1)

# MLD Information
input_path = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/"
mldpath    = input_path + "mld/" # Take from model input file, processed by prep_SSS_inputs
mldnc      = "CESM1_HTR_FULL_HMXL_NAtl.nc"
hclim      = xr.open_dataset(mldpath + "CESM1_HTR_FULL_HMXL_NAtl.nc").h.load() # [mon x ens x lat x lon]

# Qek Information
output_path_uek = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/NATL_proc/"

nc_qek_out  =  "%sCESM1_HTR_FULL_Qek_%s_NAO_%s_%s_NAtl.nc" % (outpath,varname,dampstr,rollstr) # <-- note, this seems to be the direct regression
savename_uek = "%sCESM1_HTR_FULL_Uek_NAO_%s_%s_NAtl.nc" % (outpath,dampstr,rollstr)

# CESM1 LE Inputs <End> -----------------------------------------------------




# -----------------------------------------
#%% Part 1: CALCULATE TEMPERATURE GRADIENTS
# -----------------------------------------
if calc_dT:
    #% Load the data (temperature, not anomalized)
    st   = time.time()
    ds   = xr.open_dataset(rawpath + ncname_var).load()
    print("Completed in %.2fs"%(time.time()-st))
    
    # Calculate the mean temperature for each month
    ts_monmean = ds[varname].groupby('time.month').mean('time')
    
    # Calculate dx and dy, convert to dataarray (NOTE: Need to check if lon360 is required...)
    dx,dy   = proc.calc_dx_dy(ds.lon.values,ds.lat.values)
    dx2,dy2 = proc.calc_dx_dy(ds.lon.values,ds.lat.values,centered=True)
    
    daconv   = [dx,dy,dx2,dy2]
    llcoords = {'lat':ds.lat.values,'lon':ds.lon.values,}
    da_out   = [xr.DataArray(ingrad,coords=llcoords,dims=llcoords) for ingrad in daconv]
    dx,dy,dx2,dy2 = da_out
    
    #% (1.) Temperature Gradient (Forward Difference) ------
    # Roll longitude along axis for <FORWARD> difference (gradT_x0 = T_x1 - xT0)
    dTdx = (ts_monmean.roll(lon=-1) - ts_monmean) / dx
    dTdy = (ts_monmean.roll(lat=-1) - ts_monmean) / dy
    dTdy.loc[dict(lat=dTdy.lat.values[-1])] = 0 # Set top latitude to zero (since latitude is not periodic)
    
    # Save output [ens x mon x lat x lon]
    savename  = savename_grad #"
    dsout = xr.merge([dTdx.rename('dTdx'),dTdy.rename('dTdy')])
    edict = proc.make_encoding_dict(dsout) 
    dsout.to_netcdf(savename,encoding=edict)
    print("Saved forward difference output to: %s" % savename)
    
    #% (2.) Temperature Gradient (Centered Difference) ---

    # Calculate <CENTERED> difference
    dTx2 = (ts_monmean.roll(lon=-1) - ts_monmean.roll(lon=1)) / dx2
    dTy2 = (ts_monmean.roll(lat=-1) - ts_monmean.roll(lat=1)) / dy2
    dTy2.loc[dict(lat=dTy2.lat.values[-1])] = 0 # Set top latitude to zero (since latitude is not periodic)
    
    # Save output [ens x mon x lat x lon]
    savename  = savename_grad.replace('gradT','gradT2') #savename_grad.#"%sCESM1_HTR_FULL_Monthly_gradT2_%s.nc" % (rawpath,varname)
    dsout = xr.merge([dTx2.rename('dTdx2'),dTy2.rename('dTdy2')])
    edict = proc.make_encoding_dict(dsout) 
    dsout.to_netcdf(savename,encoding=edict)
    print("Saved centered difference output to: %s" % savename)
    
    
else: # Load pre-calculated gradient files
    print("Pre-calculated gradient files will be loaded for %s." % varname)
    
if centered:
    savename  = savename_grad.replace('gradT','gradT2') #"%sCESM1_HTR_FULL_Monthly_gradT2_%s.nc" % (rawpath,varname)
else:
    savename  = savename_grad #"%sCESM1_HTR_FULL_Monthly_gradT_%s.nc" % (rawpath,varname)
ds_dT = xr.open_dataset(savename).load()

# ----------------------------------
#%% Part 2: WIND STRESS (REGRESSION)
# ----------------------------------

# Load the wind stress # [ensemble x time x lat x lon180]
# -------------------------------------------------------
# (as processed by prepare_inputs_monthly)
if calc_dtau:
    print("Loading Raw Wind Stress for calculations...")
    st          = time.time()
    taux        = xr.open_dataset(rawpath + tauxnc).load() # (ensemble: 42, time: 1032, lat: 96, lon: 89)
    tauy        = xr.open_dataset(rawpath + tauync).load()
    print("Loaded variables in %.2fs" % (time.time()-st))
    
    # Convert stress from stress on OCN on ATM --> ATM on OCN
    taux_flip   = taux.TAUX * -1
    tauy_flip   = tauy.TAUY * -1
    
    # Compute Anomalies (NOTE! They have not been detrended...)
    taux_anom   = proc.xrdeseason(taux_flip)
    tauy_anom   = proc.xrdeseason(tauy_flip)

# Note, for debugging plots of windstress, Ctrl+F <DEBUG_TAU>

#%% Compute Wind Stress regressions to NAO, if option is set
if regress_nao:
    if calc_dtau:
        print("Recalculating NAO regressions of wind stress")
        
        # Load NAO Principle Components
        dsnao                = xr.open_dataset(eofname)
        # if concat_ens: # rawpath already in eofname
        #     dsnao                = xr.open_dataset(eofname)
        # else:
        #     dsnao                = xr.open_dataset(rawpath + eofname)
        pcs                  = dsnao.pcs # [mode x mon x ens x yr]
        nmode,nmon,nens,nyr  = pcs.shape
        
        # Standardize PC
        pcstd                = pcs / pcs.std('yr')
        
        # Perform regression in a loop
        if concat_ens: # Concat Windstress and perform calculations
        
            # First, detrend the wind stress
            intaus    = [taux_anom, tauy_anom]
            taunames  = ['TAUX','TAUY']
            intaus_dt = [t - t.mean('ensemble') for t in intaus] 
            
            # Reshape and combine ens and time
            intaus_dt    = [t.transpose('ensemble','time','lat','lon') for t in intaus_dt]
            nens,ntime,nlat,nlon=intaus_dt[0].shape
            intaus_dt    = [t.data.reshape(1,nens*ntime,nlat,nlon) for t in intaus_dt]
            ntime_x_ens  = intaus_dt[0].shape[1]
            timefake     = proc.get_xryear('0000',nmon=ntime_x_ens)
            coords       = dict(ensemble=dsnao.ens.data,time=timefake,lat=intaus[0].lat,lon=intaus[0].lon)
            taus_comb    = [xr.DataArray(intaus_dt[ii],coords=coords,dims=coords,name=taunames[ii]) for ii in range(2)]
            
            # Perform Regressions of each variable
            nao_tau     = np.zeros((2,1,nlat*nlon,nmon,nmode)) * np.nan # [taux x tauy, space, month, mode]
            for tt in range(2):
                
                varin = taus_comb[tt].data
                varin = varin.reshape(1,nyr,nmon,nlat*nlon)
                e     = 0 # Only 1 ensemble, as things have been merged
                for im in range(nmon):
                    
                    # Select month and ensemble
                    pc_mon  = pcstd.isel(mon=im,ens=e).values # [mode x year]
                    var_mon = varin[e,:,im,:]
                    
                    # Get regression pattern
                    rpattern,_           = proc.regress_2d(pc_mon,var_mon,verbose=False)
                    nao_tau[tt,e,:,im,:] = rpattern.T.copy()
                    
            # Reshape and place into DataArray
            nao_tau = nao_tau.reshape(2,1,nlat,nlon,nmon,nmode)
            if correct_TAU_sign: # flip sign to be correct with EOF...
                print('\tCorrecting TAU sign...')
                nao_tau = nao_tau * -1
            ds   = intaus[0]
            cout = dict(
                        ens=pcs.ens.values,
                        lat=ds.lat.values,
                        lon=ds.lon.values,
                        mon=np.arange(1,13,1),
                        mode=pcs.mode.values,
                        )
            nao_tau_da = [xr.DataArray(nao_tau[ii],name=taunames[ii],coords=cout,dims=cout,).transpose('mode','ens','mon','lat','lon') for ii in range(2)]
            
            # Merge and save
            nao_taus   = xr.merge(nao_tau_da)
            edict      = proc.make_encoding_dict(nao_taus)
            savename = savename_naotau #"%sCESM1_HTR_FULL_Monthly_TAU_NAO_%s_%s.nc" % (rawpath,dampstr,rollstr)
            nao_taus.to_netcdf(savename,encoding=edict)
            
        else: # Note, in this approach, wind stress is not detrended...
            nens,ntime,nlat,nlon = taux.TAUX.shape
            npts                 = nlat*nlon
            
            # Loop for taus
            nao_taus = np.zeros((2,nens,nlat*nlon,nmon,nmode)) # [Taux/Tauy,space,month,mode]
            tau_anoms = [taux_anom,tauy_anom]
            for tt in range(2):
                tau_in   = tau_anoms[tt].values
                tau_in   = tau_in.reshape(nens,nyr,nmon,nlat*nlon)
                
                for e in tqdm(range(nens)):
                    for im in range(nmon):
                        # Select month and ensemble
                        pc_mon  = pcstd.isel(mon=im,ens=e).values # [mode x year]
                        tau_mon = tau_in[e,:,im,:] # [year x pts]
                        
                        # Get regression pattern
                        rpattern,_ = proc.regress_2d(pc_mon,tau_mon,verbose=False)
                        nao_taus[tt,e,:,im,:] = rpattern.T.copy()
            nao_taus = nao_taus.reshape(2,nens,nlat,nlon,nmon,nmode)
            
            if correct_TAU_sign: # flip sign to be correct with EOF...
                print('\tCorrecting TAU sign...')
                nao_taus = nao_taus * -1
            
            # Save the output
            cout = dict(
                        ens=pcs.ens.values,
                        lat=taux.lat.values,
                        lon=taux.lon.values,
                        mon=np.arange(1,13,1),
                        mode=pcs.mode.values,
                        )
            nao_taux = xr.DataArray(nao_taus[0,...],name="TAUX",coords=cout,dims=cout).transpose('mode','ens','mon','lat','lon')
            nao_tauy = xr.DataArray(nao_taus[1,...],name="TAUY",coords=cout,dims=cout).transpose('mode','ens','mon','lat','lon')
            nao_taus = xr.merge([nao_taux,nao_tauy])
            edict    = proc.make_encoding_dict(nao_taus)
            savename = savename_naotau #"%sCESM1_HTR_FULL_Monthly_TAU_NAO_%s_%s.nc" % (rawpath,dampstr,rollstr)
            nao_taus.to_netcdf(savename,encoding=edict)
    else:
        print("Loading NAO regressions of wind stress")
        savename = savename_naotau #"%sCESM1_HTR_FULL_Monthly_TAU_NAO_%s_%s.nc" % (rawpath,dampstr,rollstr)
        nao_taus = xr.open_dataset(savename)
        nao_taux = nao_taus.TAUX
        nao_tauy = nao_taus.TAUY
        
else:
    
    print("Using stdev(taux, tauy) because regress_nao is False")

# Note: See <<DEBUG_NAO_TAU>> for debugging plots



# ----------------------------------------------
#%% Part 3: Compute Ekman Velocities and Forcing
# ----------------------------------------------

# Note: Need to set a uniform dimension name system
def preproc_dimname(ds):
    if "ensemble" in list(ds.dims):
        ds = ds.rename(dict(ensemble='ens'))
    if 'month' in list(ds.dims):
        ds = ds.rename(dict(month='mon'))
    return ds

# Load mixed layer depth climatological cycle, already converted to meters
#mldpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/mld/" # Take from model input file, processed by prep_SSS_inputs
hclim     = xr.open_dataset(mldpath + mldnc).h.load() # [mon x ens x lat x lon]
hclim     = preproc_dimname(hclim) # ('ens', 'mon', 'lat', 'lon')


# First, let's deal with the coriolis parameters
llcoords  = {'lat':hclim.lat.values,'lon':hclim.lon.values,}
xx,yy     = np.meshgrid(hclim.lon.values,hclim.lat.values) 
f         = 2*omega*np.sin(np.radians(yy))
dividef   = 1/f 
dividef[np.abs(yy)<=6] = np.nan # Remove large values around equator
da_dividef = xr.DataArray(dividef,coords=llcoords,dims=llcoords)
# da_dividef.plot()

# Get gradients from above
if centered:
    dTdx = ds_dT['dTdx2']
    dTdy = ds_dT['dTdy2']
else:
    dTdx = ds_dT['dTdx']
    dTdy = ds_dT['dTdy']
    
dTdx = preproc_dimname(dTdx)
dTdy = preproc_dimname(dTdy)

if concat_ens:
    print("Just using ensemble-average h!")
    hclim = hclim.mean('ens')
    
    print("Also just using ensemble-averaged gradients")
    dTdx  = dTdx.mean('ens')
    dTdy  = dTdy.mean('ens')

#% Compute Ekman Velocities

# Try 3 different versions (none of which include monthly regressions...)
if regress_nao:
    print("Calculate Qek using NAO regressed wind stress")
    
    st = time.time()
    # Rename Dimensions
    #nao_taux = nao_taux.rename({'ens':'ensemble','mon': 'month'})
    #nao_tauy = nao_tauy.rename({'ens':'ensemble','mon': 'month'})
    nao_taux = preproc_dimname(nao_taux)
    nao_tauy = preproc_dimname(nao_tauy)
    
    # Compute velocities [m/s]
    u_ek    = (da_dividef * nao_tauy) / (rho*hclim)
    v_ek    = (da_dividef * -nao_taux) / (rho*hclim)
    
    # Compute Ekman Forcing
    if (varname == "SST" or varname == "TS") and convert_Wm2:
        print("\tConverting to Wm2!")
        q_ek1    = -1 * cp0 * (rho*hclim) * (u_ek * dTdx + v_ek * dTdy ) # W/m2
    else:
    #elif varname == "SSS" or varname == "SALT":
        print("Keeping Units as [degC/sec] or [psu/sec]") # psu/mon
        q_ek1    = -1 * (u_ek * dTdx + v_ek * dTdy )
    
    # Save Output
    dscd            = u_ek
    outcoords       = dict(mode=dscd.mode,mon=dscd.mon,ens=dscd.ens,lat=dscd.lat,lon=dscd.lon) 
    dsout           = [q_ek1,u_ek,v_ek]
    dsout_name      = ["Qek","Uek","Vek"]
    dsout_transpose = [ds.transpose('mode','mon','ens','lat','lon') for ds in dsout] # Transpose
    #dsout_transpose = [ds.rename({'ensemble':'ens','month': 'mon'}) for ds in dsout_transpose] # Rename Ens and Mon
    dsout_transpose = [dsout_transpose[ii].rename(dsout_name[ii]) for ii in range(3)] # Rename Variable
    dsout_merge     = xr.merge(dsout_transpose)
    edict           = proc.make_encoding_dict(dsout_merge)
    
    savename        = nc_qek_out #"%sCESM1_HTR_FULL_Qek_%s_NAO_%s_%s_NAtl.nc" % (outpath,varname,dampstr,rollstr)
    if concat_ens:
        savename = proc.addstrtoext(nc_qek_out,"_concatEns",adjust=-1)
    print("\tSaving Ekman Forcing to %s" % savename)
    dsout_merge.to_netcdf(savename,encoding=edict)
    
    # Also Save the Ensemble mean for Ens Mean
    if not concat_ens:
        savename_ensavg = proc.addstrtoext(savename,"_EnsAvg",adjust=-1)#"%sCESM1_HTR_FULL_Qek_%s_NAO_%s_%s_NAtl_EnsAvg.nc" % (outpath,varname,dampstr,rollstr)
        dsout_ensavg = dsout_merge.mean('ens')
        dsout_ensavg.to_netcdf(savename_ensavg,encoding=edict)
    
    print("Computed Ekman Forcing (EOF-based) in %.2fs" % (time.time()-st))
    
    # Save u_ek and v_ek
    ekman_ds = xr.merge([u_ek.rename('u_ek'),v_ek.rename('v_ek')])
    edict_ek = proc.make_encoding_dict(ekman_ds)
    savename = savename_uek#"%sCESM1_HTR_FULL_Uek_NAO_%s_%s_NAtl.nc" % (outpath,dampstr,rollstr)
    if concat_ens:
        savename = proc.addstrtoext(savename_uek,"_concatEns",adjust=-1)
    ekman_ds.to_netcdf(savename,encoding=edict_ek)
    
    # Save cropped NATl version
    if crop_sm:
        # Flip Longitude, crop region, 
        dsout_merge_lon180  = proc.lon360to180_xr(dsout_merge)
        dsout_merge_reg     = proc.sel_region_xr(dsout_merge_lon180,bbox_crop)
        savename            = nc_qek_out.replace(regstr,regstr_crop)
        if concat_ens:
            savename = proc.addstrtoext(savename,"_concatEns",adjust=-1)
        dsout_merge_reg.to_netcdf(savename,encoding=edict)
        
        
        # Save Ens Avg
        if not concat_ens:
            dsout_merge_reg_eavg = dsout_merge_reg.mean('ens')
            savename             = proc.addstrtoext(savename,"_EnsAvg",adjust=-1)
            dsout_merge_reg_eavg.to_netcdf(savename,encoding=edict)

    
else: # Standard Deviation based approach
    
    print("Calculate Qek based on standard deviation approach")
    
    # 1) Take seasonal stdv in anomalies -------
    u_ek     = (da_dividef * -tauy_anom.groupby('time.month').std('time'))/(rho * hclim)
    v_ek     = (da_dividef * taux_anom.groupby('time.month').std('time'))/(rho * hclim)
    if varname == "SST":
        q_ek1    = -1 * cp0 * (rho*hclim) * (u_ek * dTdx + v_ek * dTdy )
    elif varname == "SSS":
        q_ek1    = -1 * (u_ek * dTdx + v_ek * dTdy )
    else:
        print("%s not supported.")
    
    # 2) Tile the input
    in_tile  = [hclim,dTdx,dTdy]
    timefull = taux_anom.time.values 
    nyrs     = int(len(timefull)/12)
    out_tile = []
    for invar in in_tile:
        invar     = invar.transpose('ensemble','lat','lon','month')
        newcoords = dict(ensemble=invar.ensemble,lat=invar.lat,lon=invar.lon,time=timefull)
        invar     = np.tile(invar.values,nyrs)
        da_new    = xr.DataArray(invar,coords=newcoords,dims=newcoords)
        out_tile.append(da_new)
        print(invar.shape)
        
        
    # 3) Need monthly varying variables for all three?
    
    #%
    [hclim,dTdx,dTdy] = out_tile
    #%
    u_ek         = (da_dividef *   taux_anom)/(rho * hclim)
    v_ek         = (da_dividef * - tauy_anom)/(rho * hclim)
    q_ek2        = -1 * cp0 * (rho*hclim) * (u_ek * dTdx + v_ek * dTdy )
    q_ek2_monstd = q_ek2.groupby('time.month').std('time')
    
    
    #% Plot target point
    lonf = -30
    latf = 50
    
    mons3  = proc.get_monstr()
    fig,ax = viz.init_monplot(1,1)
    
    plot_qeks = [q_ek1,q_ek2_monstd]
    #qek_names = ["Method 1","hmean,"]
    for ii in range(2):
        plotvar = plot_qeks[ii].sel(lon=lonf,lat=latf,method='nearest').isel(ensemble=0)
        ax.plot(mons3,np.abs(plotvar),label="Method %i" % (ii+1))
    ax.legend()
    
    #% PPlot map
    
    kmonth = 1
    
    vmax=1e2
    
    bbplot = [-80,0,0,65]
    fig,axs = viz.geosubplots(1,3,figsize=(12,4.5))
    for a in range(2):
        ax = axs[a]
        ax = viz.add_coast_grid(ax,bbplot)
        
        pv  = plot_qeks[a].mean('ensemble').isel(month=kmonth)
        pv[pv>1]
        pcm = ax.pcolormesh(pv.lon,pv.lat,pv,vmin=-vmax,vmax=vmax,cmap="RdBu_r")
        fig.colorbar(pcm,ax=ax,fraction=0.025)
        ax.set_title("Method %0i" % (a+1))
        
        
    ax = axs[2]
    ax = viz.add_coast_grid(ax,bbplot)
    pv = np.abs(plot_qeks[1].mean('ensemble').isel(month=kmonth)) - np.abs((plot_qeks[0].mean('ensemble').isel(month=kmonth)))
    pcm = ax.pcolormesh(pv.lon,pv.lat,pv,vmin=-1e1,vmax=1e1,cmap="RdBu_r")
    fig.colorbar(pcm,ax=ax,fraction=0.025)
    ax.set_title("Method 2 - Method 1")
    plt.suptitle("Ens Mean Variance for Month %s" % (mons3[a]))
    # Next, create a coast mask
    
    #%% I think if we are going for physical meaning, it seems like the second method makes the most sense
    
    # Lets save the output # Should be [month x ens x lat x lon]
    dsout   = q_ek2_monstd.transpose('month','ensemble','lat','lon')
    dsout   = dsout.rename(dict(month='mon',ensemble='ens'))
    
    # Remove points with unrealistically high values (above 1000 W/m2)
    # Need to learn how to do this in xarray so I dont have to unload/reload
    varout = dsout.values
    varout = np.where(varout>1e3,np.nan,varout)
    outcoords = dict(mon=dsout.mon,ens=dsout.ens,lat=dsout.lat,lon=dsout.lon) 
    daout     = xr.DataArray(varout,coords=outcoords,dims=outcoords,name="Qek")
    edict     = {"Qek":{'zlib':True}}
    savename  = "%sCESM1_HTR_FULL_Qek_%s_monstd_NAtl.nc" % (outpath,varname)
    daout.to_netcdf(savename,encoding=edict)
    
    
    # Redo for ensemble mean
    savename2  = "%sCESM1_HTR_FULL_Qek_%s_monstd_NAtl_EnsAvg.nc" % (outpath,varname)
    daout2 = daout.mean('ens')
    daout2.to_netcdf(savename2,encoding=edict)


# NOTE: For debugging plots, see <<DEBUG_QEK_NAO>>


# ======================================================== ||| ||| ||| ||| ||| |
#%% Use this section here to compute the Ekman Advection  =====================
# ======================================================== ||| ||| ||| ||| ||| |

anomalize = False

# (1) Load Wind Stress and Anomalize

# Load the wind stress # [ensemble x time x lat x lon180]
# -------------------------------------------------------
# (as processed by prepare_inputs_monthly)
st          = time.time()
taux        = xr.open_dataset(output_path_uek + tauxnc).load() # (ensemble: 42, time: 1032, lat: 96, lon: 89)
tauy        = xr.open_dataset(output_path_uek + tauync).load()
print("Loaded variables in %.2fs" % (time.time()-st))

# Convert stress from stress on OCN on ATM --> ATM on OCN
taux_flip   = taux.TAUX * -1
tauy_flip   = tauy.TAUY * -1

# Compute Anomalies
if anomalize:
    taux_anom   = proc.xrdeseason(taux_flip)
    tauy_anom   = proc.xrdeseason(tauy_flip)
    
    # Rename Dimension
    taux_anom   = preproc_dimname(taux_anom)
    tauy_anom   = preproc_dimname(tauy_anom)
    
    # Remove Ens. Avg for detrending
    taux_anom   = taux_anom - taux_anom.mean('ens')
    tauy_anom   = tauy_anom - tauy_anom.mean('ens')
else:
    print("Data will not be anomalized")
    taux_anom   = preproc_dimname(taux_flip)
    tauy_anom   = preproc_dimname(tauy_flip)

# Load mixed layer depth climatological cycle, already converted to meters
#mldpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/mld/" # Take from model input file, processed by prep_SSS_inputs
mldnc     = "CESM1LE_HMXL_NAtl_19200101_20050101_bilinear.nc" #NOTE: this file is in cm
hclim     = xr.open_dataset(output_path_uek + mldnc).HMXL.load() # [mon x ens x lat x lon]
hclim     = preproc_dimname(hclim) # ('ens', 'mon', 'lat', 'lon')
hclim     = hclim.groupby('time.month').mean('time')
hclim     = hclim.transpose('ens','month','lat','lon')
hclim     = hclim/100 # Convert to Meters

#hclim     = hclim.transpose('ens','lat','lon')
#hclim['month'] = proc.get_xryear()
#hclim      = hclim.rename(dict(mon='time'))
#hclim     = hclim.rename({'ens':'ensemble','mon': 'month'})

# First, let's deal with the coriolis parameters
llcoords    = {'lat':hclim.lat.values,'lon':hclim.lon.values,}
xx,yy       = np.meshgrid(hclim.lon.values,hclim.lat.values) 
f           = 2*omega*np.sin(np.radians(yy))
dividef     = 1/f 
dividef[np.abs(yy)<=6] = np.nan # Remove large values around equator
da_dividef  = xr.DataArray(dividef,coords=llcoords,dims=llcoords)

# Compute Timeseries of Ekman Advection
u_ek        =  (da_dividef / rho) * (tauy_anom.groupby('time.month')/hclim)
v_ek        = -1 * (da_dividef / rho) * (taux_anom.groupby('time.month')/hclim)

# Save output
st          = time.time()
ds_out      = xr.merge([u_ek.rename("u_ek"),v_ek.rename("v_ek")])
savename    = "%sCESM1LE_uek_NAtl_19200101_20050101_bilinear.nc" % (output_path_uek)
edict       = proc.make_encoding_dict(ds_out)
ds_out.to_netcdf(savename,encoding=edict)
print("Saved Ekman Currents in %.2fs" % (time.time()-st))

# ---------------------------------------------|------------|---------------|-|
#%% Use this section to compute the Qek =============== |||||||||||||||||||||||
# ---------------------------------------------|------------|---------------|-|
## Note this currently runs on Astraeus. Can Modify to run this on stormtrack

load_qek   = True # Set to True to load precalculated output

# Load uek
savename    = "%sCESM1LE_uek_NAtl_19200101_20050101_bilinear.nc" % (output_path_uek)
ds_uek      = xr.open_dataset(savename).load() # (lat: 96, lon: 89, ens: 42, time: 1032)
ds_uek      = ds_uek.rename(dict(ens='ensemble'))

# # Preprocessing (multiply by 100 because I divided with cm earlier. Note that this has been fixed as of 2024.08.27...)
# ds_uek['u_ek'] = ds_uek['u_ek'] * 100
# ds_uek['v_ek'] = ds_uek['v_ek'] * 100

# # Anomalize # Apply this to Qek Later
def preproc(ds):
    ds = proc.xrdeseason(ds)
    ds = ds - ds.mean('ensemble')
    return ds

# Load temperature and salinity gradients 
varnames = ['SST','SSS']
#rawpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/
#rawpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"
ds_grad  = []
for varname in varnames:
    savename_grad   = "%sCESM1_HTR_FULL_Monthly_gradT2_%s.nc" % (rawpath,varname)
    ds = xr.open_dataset(savename_grad).load() # (ensemble: 42, month: 12, lat: 96, lon: 89)
    ds_grad.append(ds)
    

#%%

if load_qek:
    #% Load Qek
    qek_byvar = []
    for vv in range(2):
        varname  = varnames[vv]
        ncname = "%sCESM1LE_Qek_%s_NAtl_19200101_20050101_bilinear.nc" % (rawpath,varname)
        ds = xr.open_dataset(ncname).load().Qek
        qek_byvar.append(ds)
    
else:
    print("Recomputing Qek")
    #% Read out the needed variables, should be in units of mon/sec
    qek_byvar = []
    for vv in range(2):
        st = time.time() 
        qek = (ds_uek.u_ek.groupby('time.month') * ds_grad[vv].dTdx2 + ds_uek.v_ek.groupby('time.month') * ds_grad[vv].dTdy2)
        qek_byvar.append(qek)
        print("Computed Ekman Terms in %.2fs" % (time.time()-st))
    
    #% Save Qek (Note this is the whole term, NOT anomalized...)
    print("Saving Qek")
    for vv in range(2):
        varname  = varnames[vv]
        qek     = qek_byvar[vv]
        ncname = "%sCESM1LE_Qek_%s_NAtl_19200101_20050101_bilinear.nc" % (rawpath,varname)
        
        qek    = qek.rename("Qek")
        edict = proc.make_encoding_dict(qek)
        qek.to_netcdf(ncname,encoding=edict)
    

#%% Regress to obtain the NAO Component

load_nao    = False
# First load the NAO
rawpath_nao = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/NATL_proc/"

# Load NAO Principle Components
dsnao = xr.open_dataset(eofname)
pcs   = dsnao.pcs # [mode x mon x ens x yr]
nmode,nmon,nens,nyr  = pcs.shape

# Standardize PC
pcstd = pcs / pcs.std('yr')

# Preprocess Qek (Detrend and Deseason)
st        = time.time()
qek_byvar = [preproc(ds) for ds in qek_byvar]
print("Detrended and Deseasoned Qek in %.2fs" % (time.time()-st))

# Perform regression in a loop
qek_byvar[0] = qek_byvar[0].transpose('ensemble','time','lat','lon',)
qek_byvar[1] = qek_byvar[1].transpose('ensemble','time','lat','lon',)
nens,ntime,nlat,nlon=qek_byvar[0].shape
npts         = nlat*nlon
if concat_ens:
    qek_byvar_in = [invar.data.reshape(1,nens*ntime,nlat,nlon) for invar in qek_byvar]
    nens,ntime,nlat,nlon=qek_byvar_in[0].shape
    
    timefake     = proc.get_xryear('0000',nmon=ntime)
    coords       = dict(ensemble=dsnao.ens.data,time=timefake,lat=qek_byvar[0].lat,lon=qek_byvar[0].lon)
    qek_byvar    = [xr.DataArray(invar,coords=coords,dims=coords,name="Qek") for invar in qek_byvar_in]
else:
    qek_byvar_in = [invar.data for invar in qek_byvar]
    
if load_nao:
    print("Loading NAO regressions of Qek")
    nao_qeks = []
    for vv in range(2):
        vname       = varnames[vv]
        savename    = "%sCESM1_HTR_FULL_Qek_%s_Monthly_TAU_NAO.nc" % (rawpath,vname) # Units are psu or degC / sec
        ds = xr.open_dataset(savename).Qek.load()
        nao_qeks.append(ds)
        
    
else:
    print("Recalculating NAO regressions of Qek")
    # Perform the regression looping for each variable
    nao_qek     = np.zeros((2,nens,nlat*nlon,nmon,nmode)) # [SST/SSS,space,month,mode]
    qek_anoms   = [qek_byvar_in[0],qek_byvar_in[1]]
    for tt in range(2):
        varin   = qek_anoms[tt]#.values
        varin   = varin.reshape(nens,nyr,nmon,nlat*nlon)
        
        for e in tqdm(range(nens)):
            for im in range(nmon):
                # Select month and ensemble
                pc_mon  = pcstd.isel(mon=im,ens=e).values # [mode x year]
                var_mon= varin[e,:,im,:] # [year x pts]
                
                # Get regression pattern
                rpattern,_=proc.regress_2d(pc_mon,var_mon,verbose=False)
                nao_qek[tt,e,:,im,:] = rpattern.T.copy()
    nao_qek = nao_qek.reshape(2,nens,nlat,nlon,nmon,nmode)
    
    cout = dict(
                ens=pcs.ens.values,
                lat=ds.lat.values,
                lon=ds.lon.values,
                mon=np.arange(1,13,1),
                mode=pcs.mode.values,
                )
    
    nao_qek_sst = xr.DataArray(nao_qek[0,...],name="Qek",coords=cout,dims=cout).transpose('mode','ens','mon','lat','lon')
    nao_qek_sss = xr.DataArray(nao_qek[1,...],name="Qek",coords=cout,dims=cout).transpose('mode','ens','mon','lat','lon')
    
    nao_qeks    = [nao_qek_sst,nao_qek_sss]
    
    edict       = proc.make_encoding_dict(nao_qek_sst)
    for vv in range(2):
        vname       = varnames[vv]
        savename    = "%sCESM1_HTR_FULL_Qek_%s_Monthly_TAU_NAO.nc" % (rawpath,vname) # Units are psu or degC / sec
        if concat_ens:
            savename = proc.addstrtoext(savename,"_concatEns",adjust=-1)
        nao_qeks[vv].to_netcdf(savename,encoding=edict)




# -----------------------------------------------------------------------------  
#%% (4) Perform EOF filtering and correction (copied from correct_eof_forcing_SSS)
# -----------------------------------------------------------------------------

# Indicate Filtering Options
eof_thres   = 0.90
bbox_crop   = [-90,0,0,90]
ensavg_only = True #  Set to True to compute only for the ensemble average values. 
varnames    = ["SST","SSS"]

if concat_ens:
    ensavg_only = True # This is automatically set to true because there is only 1 ensemble member

# Indicate the what to load
DirReg = False
if DirReg:
    method_out = "DirReg" #DirReg for Direction Regression of Qek SST onto NAO
    
    # Load NAO Qek (Note, this is in degC/sec or psu/sec!!)
    ncnames  = ["%sCESM1_HTR_FULL_Qek_%s_Monthly_TAU_NAO.nc" % (rawpath,vv) for vv in varnames]
    nao_qeks = [xr.open_dataset(nc).load() for nc in ncnames] 
    
    ncnames_out     = [
        "CESM1_HTR_FULL_Qek_SST_NAO_DirReg_NAtl.nc", #DirReg for Direction Regression of Qek SST onto NAO
        "CESM1_HTR_FULL_Qek_SSS_NAO_DirReg_NAtl.nc",
        ]
    
else:
    method_out = "TauReg"
    
    fpath   = input_path + 'forcing/'
    ncnames = ["%sCESM1_HTR_FULL_Qek_%s_NAO_nomasklag1_nroll0_NAtl_concatEns.nc" % (fpath,vv) for vv in varnames]
    
    # Load NAO Qek (Note, this is in degC/sec or psu/sec!!)
    # Load NAO Qek (NOTE: if convert_Wm2 is True, then SST might be in W/m2)
    nao_qeks = [xr.open_dataset(nc).Qek.load() for nc in ncnames] 
    

# Indicate Output Names
ncnames_out     = ["%sCESM1_HTR_FULL_Qek_%s_NAO_%s_NAtl.nc" % (outpath,vv,method_out) for vv in varnames]
if concat_ens:
    ncnames_out = [proc.addstrtoext(sn,"_concatEns",adjust=-1) for sn in ncnames_out]

# Load NAO Principle Components
dsnao = xr.open_dataset(eofname).load()
varexp_in   = dsnao.varexp.transpose('mode','ens','mon').data#.mean('ens') # (mode: 86, ens : 42, mon: 12, ens: 42)

# Get the EOF values
ds_eofraw   = nao_qeks
if DirReg: # Note, here correction is performed relative to full Qek' term... (not just NAO component)
    print("Correcting to the anomalous ekman forcing (including non-EOF components)")
    ds_std      = [ds.groupby('time.month').std('time').rename(dict(ensemble='ens',month='mon')) for ds in qek_byvar]
else:
    print("Correcting based on sqrt summed sq or EOF coefficients")
    # Correct to full stddev of NAO regression
    def sqrtsumsq(ds):
        return np.sqrt((ds**2).sum('mode'))
    ds_std = [sqrtsumsq(ds) for ds in nao_qeks]
    
# get other dims
vnames      = ['Qek',"Qek"] # Same name to fit loading convenience
nvars       = len(vnames)
lat_out     = ds_eofraw[0].lat
lon_out     = ds_eofraw[0].lon

for vv in range(nvars):
    
    # Index variables, convert to np array [ens x mon x lat x lon]
    eofvar_in       = ds_eofraw[vv].transpose('mode','ens','mon','lat','lon').values # (86, 42, 12, 96, 89)
    monvar_full     = ds_std[vv].transpose('ens','mon','lat','lon').values #  (42, 12, 96, 89)
    
    corr_check = []
    eof_check  = []
    if ensavg_only: # Take Ensemble Average of values and just compute
        
        print("Taking Ensemble Average First, then apply filter + correction")
        eofvar_in       = np.nanmean(eofvar_in,1)     # (86, 12, 96, 89)
        monvar_full     = np.nanmean(monvar_full,0)   # (12, 96, 89)
        varexp_eavg     = np.nanmean(varexp_in,1) # 
        
        # Perform Filtering
        eofs_filtered,varexp_cumu,nmodes_needed,varexps_filt=proc.eof_filter(eofvar_in,varexp_eavg,
                                                            eof_thres,axis=0,return_all=True)
        
        # Compute Stdev of EOFs
        eofs_std = np.sqrt(np.sum(eofs_filtered**2,0)) # [Mon x Lat x Lon]
        
        # Compute pointwise correction
        correction_diff = monvar_full - eofs_std
        
        # Prepare to Save -------------------------
        
        corcoords     = dict(mon=np.arange(1,13,1),lat=lat_out,lon=lon_out)
        eofcoords     = dict(mode=ds_eofraw[0].mode,mon=np.arange(1,13,1),lat=lat_out,lon=lon_out)
        
        da_correction = xr.DataArray(correction_diff,coords=corcoords,dims=corcoords,name="correction_factor")
        da_eofs_filt  = xr.DataArray(eofs_filtered,coords=eofcoords,dims=eofcoords  ,name=vnames[vv])
    
        ds_out        = xr.merge([da_correction,da_eofs_filt])
        edict         = proc.make_encoding_dict(ds_out)
        
        corr_check.append(da_correction)
        eof_check.append(da_eofs_filt)
        
        # Save for all ensemble members
        savename       = proc.addstrtoext(ncnames[vv],"_corrected",adjust=-1)
        if not concat_ens:
            savename       = proc.addstrtoext(savename,"_EnsAvgFirst",adjust=-1)
        ds_out.to_netcdf(savename,encoding=edict)
        print("Saved output to %s" % savename)
        
    else: # Loop computation for each ensemble member
        print("Apply filter + correction to each ensemble member...")
        
        # Repeat for each ensemble member
        nens            = monvar_full.shape[0]
        filtout_byens = []
        for e in range(nens):
            
            # Perform Filtering
            filtout=proc.eof_filter(eofvar_in[:,e,...],varexp_in[:,e,:],
                                                               eof_thres,axis=0,return_all=True)
            #eofs_filtered,varexp_cumu,nmodes_needed,varexps_filt = filtout
            filtout_byens.append(filtout)
        eofs_filtered = np.array([arr[0] for arr in filtout_byens]) # (42, 86, 12, 96, 89)
        
        # Compute stdev of EOFs
        eofs_std = np.sqrt(np.sum(eofs_filtered**2,1)) # [Ens x Mon x Lat x Lon]
        
        # Compute pointwise correction
        correction_diff = monvar_full - eofs_std
        
        # Prepare to Save -------------------------
        corcoords     = dict(ens=ds_std[0].ens,mon=np.arange(1,13,1),lat=lat_out,lon=lon_out)
        eofcoords     = dict(mode=ds_eofraw[0].mode,ens=ds_std[0].ens,mon=np.arange(1,13,1),lat=lat_out,lon=lon_out)
        
        da_correction = xr.DataArray(correction_diff,coords=corcoords,dims=corcoords,name="correction_factor")
        eofs_filtered = eofs_filtered.transpose(1,0,2,3,4)
        da_eofs_filt  = xr.DataArray(eofs_filtered,coords=eofcoords,dims=eofcoords  ,name=vnames[vv])
    
        ds_out        = xr.merge([da_correction,da_eofs_filt])
        edict         = proc.make_encoding_dict(ds_out)
    
        # Save for all ensemble members
        savename       = proc.addstrtoext(ncnames_out[vv],"_corrected",adjust=-1)
        ds_out.to_netcdf(savename,encoding=edict)
        
        # Save Ens Avg
        savename_emean = proc.addstrtoext(savename,"_EnsAvg",adjust=-1)
        ds_out_ensavg  = ds_out.mean('ens')
        ds_out_ensavg.to_netcdf(savename_emean,encoding=edict)
        
        print("Saved output to %s" % savename_emean)
        
    # if crop_sm:
    #     print("Cropping to region %s" % (regstr_crop))
    #     ds_out = proc.lon360to180_xr(ds_out)
        
    #     ds_out_reg = proc.sel_region_xr(ds_out,bbox_crop)
    #     savename_reg = proc.addstrtoext(ncnames[v],"_corrected",adjust=-1).replace(regstr,regstr_crop)
    #     ds_out_reg.to_netcdf(savename_reg,encoding=edict)
        
    #     savename_emean = proc.addstrtoext(savename_reg,"_EnsAvg",adjust=-1)
    #     ds_out_reg_ensavg  = ds_out_reg.mean('ens')
    #     ds_out_reg_ensavg.to_netcdf(savename_emean,encoding=edict)
    #     print("Saved Ens Avg. Cropped Output to %s" % savename_emean)
        
#%% Check it
  
def stdsqsum(ds,axis=0):
    return np.sqrt(np.sum(ds**2,axis))
lonf = -36
latf = 50
vv = 0
eofspt = proc.selpt_ds(eof_check[vv],lonf,latf)
eofspt = stdsqsum(eofspt)

corrpt   = np.sqrt(proc.selpt_ds(corr_check[vv]**2,lonf,latf))
monvarpt = proc.selpt_ds(ds_std[vv],lonf,latf).mean('ens')

plt.plot(eofspt,label="EOF Forcing",color="b")
plt.plot(monvarpt,label="Monthly Variance",color="k")
plt.plot(corrpt,label="Correction",color="red")
plt.plot(corrpt+eofspt,label="Final",color="cyan",ls='dashed')

plt.legend()
plt.show()


# <0> Visualization and Scrap =================================================
#%% Look at NAO component
# -----------------------------------------------------------------------------

# Visualize regression patterns for selected modes
selmodes = [0,1,2]
e=0
m=0

fig,axs = plt.subplots(1,3,subplot_kw={'projection':ccrs.PlateCarree()})

for ii in range(3):
    ax = axs[ii]
    ax.coastlines()
    ax.set_extent(bboxplot)
    plotvar = nao_qek_sss.isel(mode=selmodes[ii],ens=e,mon=m) * dtmon
    pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,vmin=-.00001,vmax=.00001)
    fig.colorbar(pcm,ax=ax,fraction=0.045,pad=0.01)
    ax.set_title("Mode %i" % (selmodes[ii]+1))

plt.show()


#%% Load get additional things for plotting

def plot_vel(plotu,plotv,qint,ax,proj=ccrs.PlateCarree(),scale=1,c='k'):
    lon     = plotu.lon.data
    lat     = plotu.lat.data
    qv      = ax.quiver(lon[::qint],lat[::qint],
                        plotu.data[::qint,::qint],plotv.data[::qint,::qint],
                        transform=proj,scale=scale,color=c)
    return qv
    

#%% Verify Instantaneous Qek (along with instantaneous vectors)
# Eventually add in gradients to double check

e           = 0
t           = 0
vv          = 0
vlms        = [[-.1,0.1],[-.01,.01]]
dtmon       = 3600*24*30
bboxplot    = [-80,0,20,65]

# Initialize the map
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()})
ax.coastlines()
ax.set_extent(bboxplot)


# Plot the Qek Forcing
plotvar = qek_byvar[vv].isel(ensemble=e,time=t) * dtmon
pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,vmin=vlms[vv][0],vmax=vlms[vv][1])


# Plot the quivers
qint=2
plotu = ds_uek.u_ek.isel(ensemble=e,time=t)
plotv = ds_uek.v_ek.isel(ensemble=e,time=t)
#qv      = plot_vel(plotu,plotv,2,ax=ax,scale=0.5)
lon     = plotu.lon.data
lat     = plotu.lat.data
qv      = ax.quiver(lon[::qint],lat[::qint],
                    plotu.data[::qint,::qint],plotv.data[::qint,::qint],)

fig.colorbar(pcm,ax=ax,fraction=0.045,pad=0.01)

plt.show()

#%% Next Section, Compare Ekman forcing (tau nao) versus Ekman forcing (direct regression)

# Load Direct Regression onto NAO
nao_qeks = []
for vv in range(2):
    vname       = varnames[vv]
    savename    = "%sCESM1_HTR_FULL_Qek_%s_Monthly_TAU_NAO.nc" % (rawpath,vname) # Units are psu or degC / sec
    ds = xr.open_dataset(savename).load()
    nao_qeks.append(ds)
    
nao_qeks_ensavg= [ds.Qek.mean('ens') for ds in nao_qeks]

# Load Qek from TAU-Regressed NAO
tau_qeks = []
for vv in range(2):
    varname       = varnames[vv]
    nc = "%sCESM1_HTR_FULL_Qek_%s_NAO_%s_%s_NAtl_EnsAvg.nc" % (outpath,varname,dampstr,rollstr)
    ds = xr.open_dataset(nc).load()
    tau_qeks.append(ds)

#%% Plot Differences

nmode       = 0
im          = 0
vlims_nao   = [[-.0001,.0001],[-.0001,.0001]]
vlims_tau   = [[-20,20],[-1e-9,1e-9]]

# Initialize the map
fig,axs = plt.subplots(2,2,subplot_kw={'projection':ccrs.PlateCarree()})

for vv in range(2):
    
    for ii in range(2):
        
        if ii == 0:
            title       = "Regress Tau"
            plotvar     = tau_qeks[vv].isel(mode=nmode,mon=im).Qek
            vlims = vlims_tau
            
            
        else:
            title="Regress Qek"
            plotvar    = nao_qeks_ensavg[vv].isel(mode=nmode,mon=im) * dtmon #.Qek * dtmon
            vlims = vlims_nao
            
        

        ax = axs[vv,ii]
        
        ax.coastlines()
        ax.set_extent(bboxplot)
        if vv == 0:
            ax.set_title(title)
        if ii == 0:
            viz.add_ylabel(varnames[vv],ax=ax)
            
            
        pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,vmin=vlims[vv][0],vmax=vlims[vv][1])
        fig.colorbar(pcm,ax=ax,fraction=0.045,pad=0.01)
        
plt.show()


#%% For Debugging, Check the difference of doing ens avg corrextion
# or correcting all the ensemble members individually then taking ens avg

# Run the internal section of the loop ( needed varexp_in, eofvar_in)


# First, compute by ensemble member,,,,
nens            = monvar_full.shape[0]
filtout_byens = []
for e in range(nens):
    
    # Perform Filtering
    filtout=proc.eof_filter(eofvar_in[:,e,...],varexp_in[:,e,:],
                                                       eof_thres,axis=0,return_all=True)
    eofs_filtered,varexp_cumu,nmodes_needed,varexps_filt = filtout
    filtout_byens.append(filtout)
eofs_filtered_allens = np.array([arr[0] for arr in filtout_byens])
varexp_cumu_allens   = np.array([arr[1] for arr in filtout_byens]) #  (42, 86, 12)


# Next, take ensemble average then correct
eofvar_in   = eofvar_in.mean('ens')     # (86, 12, 96, 89)
monvar_full = monvar_full.mean('ens')   # (12, 96, 89)

# Perform Filtering
eofs_filtered,varexp_cumu,nmodes_needed,varexps_filt=proc.eof_filter(eofvar_in,varexp_in,
                                                    eof_thres,axis=0,return_all=True)

#% Make a plot of cuulative variance explained
im = 0
fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12,4.5))
for e in range(nens):
    ax.plot(varexp_cumu_allens[e,:,im],alpha=0.1)
ax.plot(varexp_cumu_allens[:,:,im].mean(0),alpha=1,color="k",label="Ens Avg Last")


ax.plot(varexp_cumu[:,im],label="Ens Avg First",color='red',ls='dashed')
ax.legend()
ax.set_ylim([0.25,1.05])
ax.set_xlim([0,85])
plt.show()



#%% << DEBUGGING CENTER >> =====================================================

# Declare some common variables
bbox    = [-80,0,0,65]

#%% <<DEBUG_QEK_NAO>> check the Qek Computed with Ekman Currents and NAO
"""
What you need
q_ek1
u_ek
v_ek
nao_taux
nao_taoy

"""

iens  = 0
imon  = 0
imode = 0

fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},
                      constrained_layout=True)
ax = viz.init_map(bbox,ax=ax)

plotvar = q_ek1.isel(ens=iens,mon=imon,mode=imode)
pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,vmin=-40,vmax=40,cmap='cmo.balance')
cb = viz.hcbar(pcm)


# Plot the Ekman Currents
qint  = 2
plotu = u_ek.isel(ens=iens,mon=imon,mode=imode)
plotv = v_ek.isel(ens=iens,mon=imon,mode=imode)
#qv      = plot_vel(plotu,plotv,2,ax=ax,scale=0.5)
lon     = plotu.lon.data
lat     = plotu.lat.data
qv      = ax.quiver(lon[::qint],lat[::qint],
                    plotu.data[::qint,::qint],plotv.data[::qint,::qint],color='blue')


# Plot Wind Stress
qint  = 2
plotu = nao_taux.isel(ens=iens,mon=imon,mode=imode)
plotv = nao_tauy.isel(ens=iens,mon=imon,mode=imode)
#qv      = plot_vel(plotu,plotv,2,ax=ax,scale=0.5)
lon     = plotu.lon.data
lat     = plotu.lat.data
qv      = ax.quiver(lon[::qint],lat[::qint],
                    plotu.data[::qint,::qint],plotv.data[::qint,::qint],color='dimgray')



plt.show()

#%% << DEBUG_TAU >> Plot the MEAN taux and tauy

itime = 0
iens  = 0


fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},
                      constrained_layout=True)
ax     = viz.init_map(bbox,ax=ax)

# Plot the tau
qint  = 2
plotu = taux_flip.isel(ensemble=iens,time=itime)
plotv = tauy_flip.isel(ensemble=iens,time=itime)
lon     = plotu.lon.data
lat     = plotu.lat.data
ubar  = np.sqrt(plotu**2 + plotv**2).data
# qv      = ax.quiver(lon[::qint],lat[::qint],
#                     plotu.data[::qint,::qint],plotv.data[::qint,::qint],color='gray')

slns      = ax.streamplot(lon[::qint],lat[::qint],
                    plotu.data[::qint,::qint],plotv.data[::qint,::qint],
                    color=ubar[::qint,::qint],cmap='cmo.curl')


plt.show()

#%% <<DEBUG_NAO_TAU>> Check NAO Wind Stress with EOF Pattern

iens    = 0
imon    = 0
imode   = 0


fig,ax  = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},
                      constrained_layout=True)
ax      = viz.init_map(bbox,ax=ax)

# Plot EOF Forcing (Fprime)
plotvar = eofs
eofs    = dsnao.eofs.isel(ens=iens,mon=imon,mode=imode)
pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,vmin=-40,vmax=40,cmap='cmo.balance')
cb      = viz.hcbar(pcm)

# Plot Wind Stress
qint    = 2
plotu   = nao_taux.isel(ens=iens,mon=imon,mode=imode)
plotv   = nao_tauy.isel(ens=iens,mon=imon,mode=imode)
lon     = plotu.lon.data
lat     = plotu.lat.data
qv      = ax.quiver(lon[::qint],lat[::qint],
                    plotu.data[::qint,::qint],plotv.data[::qint,::qint],color='dimgray')

plt.show()

# --------------------  
#%% Double Check <Uek>
# --------------------
bboxplot = bbox 
e        = 2
t        = 48


fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},constrained_layout=True,figsize=(12,8.5))
ax.coastlines()
ax.set_extent(bboxplot)
                      
    
# Plot the quivers
qint  = 2
plotu = ds_uek.u_ek.isel(ensemble=e,time=t)
plotv = ds_uek.v_ek.isel(ensemble=e,time=t)
#qv      = plot_vel(plotu,plotv,2,ax=ax,scale=0.5)
lon     = plotu.lon.data
lat     = plotu.lat.data
qv      = ax.quiver(lon[::qint],lat[::qint],
                    plotu.data[::qint,::qint],plotv.data[::qint,::qint],)
qk = ax.quiverkey(qv,.0,1,0.1,r"0.1 $\frac{m}{s}$",fontproperties=dict(size=10))

pm      = np.sqrt((plotu**2 + plotv**2))
pcm     = ax.pcolormesh(pm.lon,pm.lat,pm,zorder=-1)
fig.colorbar(pcm,ax=ax)

plt.show()