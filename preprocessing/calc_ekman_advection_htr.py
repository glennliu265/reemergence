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
(3) Compute Ekman Velocities and Forcing
(4) Compute full Ekman Term (Qek), Regress to NAO

Script History
------------------------
 - Copied calc_ekman_advection from stochmod on 2024.02.06
 - Calculate Ekman Advection for the corresponding EOFs and save
 - Uses variables processed by investigate_forcing.ipynb

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

#%% User Edits

# Set Constants
omega = 7.2921e-5 # rad/sec
rho   = 1026      # kg/m3
cp0   = 3996      # [J/(kg*C)]
mons3 = proc.get_monstr()#('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')

# Variable Input options -----------------------

# CESM1 LE Regrid 5deg Inputs -----------------------
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

# End -----------------------------------
# CESM1 LE Inputs -----------------------
varname      = "SST"
rawpath         = rawpath # Read from above
ncname_var      = "CESM1LE_%s_NAtl_19200101_20050101_bilinear.nc" % varname
savename_grad   = "%sCESM1_HTR_FULL_Monthly_gradT_%s.nc" % (rawpath,varname)

# Wind Stress Information
tauxnc = "CESM1LE_TAUX_NAtl_19200101_20050101_bilinear.nc"
tauync = "CESM1LE_TAUY_NAtl_19200101_20050101_bilinear.nc"

# EOF Information
dampstr    = "nomasklag1"
rollstr    = "nroll0"
eofname    = "%sEOF_Monthly_NAO_EAP_Fprime_%s_%s_NAtl.nc" % (rawpath,dampstr,rollstr)
savename_naotau = "%sCESM1_HTR_FULL_Monthly_TAU_NAO_%s_%s.nc" % (rawpath,dampstr,rollstr)

# MLD Information
input_path = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/"
mldpath    = input_path + "mld/" # Take from model input file, processed by prep_SSS_inputs
mldnc      = "CESM1_HTR_FULL_HMXL_NAtl.nc"
hclim      = xr.open_dataset(mldpath + "CESM1_HTR_FULL_HMXL_NAtl.nc").h.load() # [mon x ens x lat x lon]

# Qek Information
output_path_uek = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/NATL_proc/"
nc_qek_out  =  "%sCESM1_HTR_FULL_Qek_%s_NAO_%s_%s_NAtl.nc" % (outpath,varname,dampstr,rollstr)
savename_uek = "%sCESM1_HTR_FULL_Uek_NAO_%s_%s_NAtl.nc" % (outpath,dampstr,rollstr)
#----------------------------------------------

# Calculation Options
centered    = True  # Set to True to load centered-difference temperature

calc_dT     = True # Set to True to recalculate temperature gradients (Part 1)
calc_dtau   = False # Set to True to perform wind-stress regressions to PCs (Part 2)

calc_qek    = True  # set to True to calculate ekman forcing 
debug       = True  # Set to True to visualize for debugging

regress_nao = True # Set to True to compute Qek based on wind stress regressed to NAO. Otherwise, use stdev(taux/tauy anoms)


crop_sm     = True
bbox_crop   = [-90,0,0,90]
regstr_crop = "NAtl"

correction     = True # Set to True to Use Fprime (T + lambda*T) instead of Qnet
correction_str = "_Fprime_rolln0" # Add this string for loading/saving


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
    print("Pre-calculated gradient files will be loaded.")
    
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
st          = time.time()
taux        = xr.open_dataset(rawpath + tauxnc).load() # (ensemble: 42, time: 1032, lat: 96, lon: 89)
tauy        = xr.open_dataset(rawpath + tauync).load()
print("Loaded variables in %.2fs" % (time.time()-st))

# Convert stress from stress on OCN on ATM --> ATM on OCN
taux_flip   = taux.TAUX * -1
tauy_flip   = tauy.TAUY * -1

# Compute Anomalies
taux_anom   = proc.xrdeseason(taux_flip)
tauy_anom   = proc.xrdeseason(tauy_flip)

#%% Compute Wind Stress regressions to NAO, if option is set
if regress_nao:
    if calc_dtau:
        print("Recalculating NAO regressions of wind stress")
        # Load NAO Principle Components
        dsnao = xr.open_dataset(rawpath + eofname)
        pcs   = dsnao.pcs # [mode x mon x ens x yr]
        nmode,nmon,nens,nyr  =pcs.shape
        
        # Standardize PC
        pcstd = pcs / pcs.std('yr')
        
        # Perform regression in a loop
        nens,ntime,nlat,nlon=taux.TAUX.shape
        npts     = nlat*nlon
        
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
                    tau_mon= tau_in[e,:,im,:] # [year x pts]
                    
                    # Get regression pattern
                    rpattern,_=proc.regress_2d(pc_mon,tau_mon,verbose=False)
                    nao_taus[tt,e,:,im,:] = rpattern.T.copy()
        nao_taus = nao_taus.reshape(2,nens,nlat,nlon,nmon,nmode)
        
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


# ----------------------------
#%% Part 3: Compute Ekman Velocities
# ----------------------------

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

#hclim     = hclim.rename({'ens':'ensemble','mon': 'month'})

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
    if varname == "SST" or varname == "TS":
        q_ek1    = -1 * cp0 * (rho*hclim) * (u_ek * dTdx + v_ek * dTdy ) # W/m2
    elif varname == "SSS" or varname == "SALT":
        print("Doing Simpler Conversion for SSS") # psu/mon
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
    dsout_merge.to_netcdf(savename,encoding=edict)
    
    # Redo for Ens Mean
    savename_ensavg = proc.addstrtoext(savename,"_EnsAvg",adjust=-1)#"%sCESM1_HTR_FULL_Qek_%s_NAO_%s_%s_NAtl_EnsAvg.nc" % (outpath,varname,dampstr,rollstr)
    dsout_ensavg = dsout_merge.mean('ens')
    dsout_ensavg.to_netcdf(savename_ensavg,encoding=edict)
    
    print("Computed Ekman Forcing (EOF-based) in %.2fs" % (time.time()-st))
    
    # Save u_ek and v_ek
    ekman_ds = xr.merge([u_ek.rename('u_ek'),v_ek.rename('v_ek')])
    edict_ek = proc.make_encoding_dict(ekman_ds)
    savename = savename_uek#"%sCESM1_HTR_FULL_Uek_NAO_%s_%s_NAtl.nc" % (outpath,dampstr,rollstr)
    ekman_ds.to_netcdf(savename,encoding=edict_ek)
    
    # Save cropped NATl version
    if crop_sm:
        # Flip Longitude, crop region, 
        dsout_merge_lon180  = proc.lon360to180_xr(dsout_merge)
        dsout_merge_reg     = proc.sel_region_xr(dsout_merge_lon180,bbox_crop)
        savename            = nc_qek_out.replace(regstr,regstr_crop)
        dsout_merge_reg.to_netcdf(savename,encoding=edict)
        
        # Save Ens Avg
        dsout_merge_reg_eavg = dsout_merge_reg.mean('ens')
        savename             = proc.addstrtoext(savename,"_EnsAvg",adjust=-1)
        dsout_merge_reg_eavg.to_netcdf(savename,encoding=edict)
        
        #ds_out_ensavg_reg = 
    
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

load_qek   = False # Set to True to load precalculated output


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
    qek_anoms   = [qek_byvar[0],qek_byvar[1]]
    for tt in range(2):
        varin   = qek_anoms[tt].values
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
                lat=qek_byvar[0].lat.values,
                lon=qek_byvar[0].lon.values,
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
        
        nao_qeks[vv].to_netcdf(savename,encoding=edict)

# -----------------------------------------------------------------------------  
#%% Perform EOF filtering and correction (copied from correct_eof_forcing_SSS)
# -----------------------------------------------------------------------------

# Indicate Filtering Options
eof_thres   = 0.90
bbox_crop   = [-90,0,0,90]
ensavg_only = True #  Set to True to compute only for the ensemble average values
varnames    = ["SST","SSS"]
ncnames     = [
    "CESM1_HTR_FULL_Qek_SST_NAO_DirReg_NAtl.nc", #DirReg for Direction Regression of Qek SST onto NAO
    "CESM1_HTR_FULL_Qek_SSS_NAO_DirReg_NAtl.nc",
    ]
ncnames     = [outpath + nc for nc in ncnames]

varexp_in   = dsnao.varexp.transpose('mode','ens','mon').data#.mean('ens') # (mode: 86, ens : 42, mon: 12, ens: 42)
vnames      = ['Qek',"Qek"] # Same name to fit loading convenience
ds_eofraw   = nao_qeks
ds_std      = [ds.groupby('time.month').std('time').rename(dict(ensemble='ens')) for ds in qek_byvar]
nvars       = len(vnames)

lat_out     = ds_eofraw[0].lat
lon_out     = ds_eofraw[0].lon

for vv in range(nvars):
    
    # Index variables, convert to np array
    eofvar_in       = ds_eofraw[vv].transpose('mode','ens','mon','lat','lon').values # (86, 42, 12, 96, 89)
    monvar_full     = ds_std[vv].transpose('ens','month','lat','lon').values #  (42, 12, 96, 89)
    
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
        
        # Save for all ensemble members
        savename       = proc.addstrtoext(ncnames[vv],"_corrected",adjust=-1)
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
        savename       = proc.addstrtoext(ncnames[vv],"_corrected",adjust=-1)
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

# --------------------  
#%% Double Check <Uek>
# --------------------
bboxplot = [-80,0,20,65]
e = 2
t = 48


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
    


#%% Even more Scrap Below, not sure what it's for....

#%%

taux_pt = taux_anom.sel(lon=lonf,lat=latf,method='nearest').isel(ens=0)
hpt     = hclim.sel(lon=lonf,lat=latf,method='nearest').isel(ens=0)

taux_pt.groupby('time.month') / hpt
 
-1.38243343e-06

tx = -0.01305926#-1.38243343e-06
h  = 9446.57515741


#%%
# #%% OLD SCRIPT BELOW
# #%%

# #% Load the wind stress and the PCs to prepare for regression
# # -----------------------------------------------------------
# # This was calculated in NHFLX_EOF_monthly.py
# N_mode = 200

# # Load the PCs
# savename = "%sNHFLX_FULL-PIC_%sEOFsPCs_lon260to20_lat0to65.npz" % (rawpath,N_mode)
# if correction:
#     savename = proc.addstrtoext(savename,correction_str)

# ld        = np.load(savename,allow_pickle=True)
# pcall     = ld['pcall'] # [PC x MON x TIME]
# eofall    = ld['eofall']
# eofslp    = ld['eofslp']
# varexpall = ld['varexpall']
# lon       = ld['lon']
# lat       = ld['lat']

# # Flip signs
# spgbox     = [-60,20,40,80]
# eapbox     = [-60,20,40,60] # Shift Box west for EAP
# N_modeplot = 5              # Just flip the first few
# for N in tqdm(range(N_modeplot)):
#     if N == 1:
#         chkbox = eapbox # Shift coordinates west
#     else:
#         chkbox = spgbox
#     for m in range(12):
        
#         sumflx = proc.sel_region(eofall[:,:,[m],N],lon,lat,chkbox,reg_avg=True)
#         sumslp = proc.sel_region(eofslp[:,:,[m],N],lon,lat,chkbox,reg_avg=True)
        
#         if sumflx > 0:
#             print("Flipping sign for NHFLX, mode %i month %i" % (N+1,m+1))
#             eofall[:,:,m,N]*=-1
#             pcall[N,m,:] *= -1
#         if sumslp > 0:
#             print("Flipping sign for SLP, mode %i month %i" % (N+1,m+1))
#             eofslp[:,:,m,N]*=-1

# if calc_dtau:
    
#     # Load each wind stress component [yr mon lat lon]
#     st   = time.time()
#     dsx  = xr.open_dataset(rawpath+"../CESM_proc/TAUX_PIC_FULL.nc")
#     taux = dsx.TAUX.values
#     dsx  = xr.open_dataset(rawpath+"../CESM_proc/TAUY_PIC_FULL.nc")
#     tauy = dsx.TAUY.values
#     print("Loaded wind stress data in %.2fs"%(time.time()-st))

#     # Convert stress from stress on OCN on ATM --> ATM on OCN
#     taux*= -1
#     tauy*= -1
    
#     #% Preprocess Wind Stress Variables
#     # ---------------------------------
#     takeanom = False 
    
#     fullx = taux.copy()
#     fully = tauy.copy()
        
#     nyr,_,nlat,nlon = taux.shape
#     taux = taux.reshape(nyr,12,nlat*nlon) # Combine space
#     tauy = tauy.reshape(nyr,12,nlat*nlon)
    
#     if takeanom:
#         taux = taux - taux.mean(1)[:,None,:]
#         tauy = tauy - tauy.mean(1)[:,None,:]



#     #% Regress wind stress components to NHFLX PCs
#     # ---------------------------------
#     taux_pat = np.zeros((nlat*nlon,12,N_mode))
#     tauy_pat = np.zeros((nlat*nlon,12,N_mode))
#     for m in tqdm(range(12)):
        
        
#         tx_in = taux[:,m,:]
#         ty_in = tauy[:,m,:]
#         pc_in = pcall[:,m,:]
#         pcstd = pc_in / pc_in.std(1)[:,None] # Standardize in time dimension
        
#         eof_x,_ = proc.regress_2d(pcstd,tx_in)
#         eof_y,_ = proc.regress_2d(pcstd,ty_in)
        
#         taux_pat[:,m,:] = eof_x.T
#         tauy_pat[:,m,:] = eof_y.T
    
#     # Reshape, postprocess
#     procvars = [taux_pat,tauy_pat]
#     fin    = []
#     for invar in procvars:
        
#         # Reshape things for more processing
#         invar = invar.reshape(nlat,nlon,12*N_mode) # Make 3D
#         invar = invar.transpose(1,0,2) # [Lon x lat x otherdims]
        
#         # Flip to degreeseast/west
#         _,outvar = proc.lon360to180(lon360,invar)
    
#         # Reshape variable
#         outvar = outvar.reshape(nlon,nlat,12,N_mode)
#         fin.append(outvar)

    
#     # Unflipped variable
#     taux_pat = taux_pat.reshape(nlat,nlon,12,N_mode)
#     tauy_pat = tauy_pat.reshape(nlat,nlon,12,N_mode)
    
#     taux_pat_fin,tauy_pat_fin = fin


#     # #%% Flip EOFs (again)
    
#     # spgbox = [-60,20,40,80]
#     # N_modeplot = 5
    
#     # for N in tqdm(range(N_modeplot)):
#     #     for m in range(12):
            
#     #         sumflx = proc.sel_region(eofall[:,:,[m],N],lon180,lat,spgbox,reg_avg=True)
#     #         #sumslp = proc.sel_region(eofslp[:,:,[m],N],lon180,lat,spgbox,reg_avg=True)
#     #         if sumflx > 0:
                
#     #             print("Flipping sign for NHFLX, mode %i month %i" % (N+1,m+1))
                
#     #             eofall[:,:,m,N] *=-1
#     #             pcall[N,m,:]    *= -1
#     #             eofslp[:,:,m,N] *=-1
#     #             taux_pat_fin[:,:,m,N] *=-1
#     #             tauy_pat_fin[:,:,m,N] *=-1

#     #% Save the files
#     # ---------------------------------
#     # Save output...
#     savename = "%sFULL-PIC_Monthly_NHFLXEOF_TAUX_TAUY_centered%i.npz" % (rawpath,centered)
#     if correction:
#         savename = proc.addstrtoext(savename,correction_str)
#     np.savez(savename,**{
#         'taux':taux_pat_fin,
#         'tauy':tauy_pat_fin,
#         'lon':lon180,
#         'lat':lat})
#     print("Saved wind-stress regression output to: %s" % savename)


# #% Load the files otherwise
# else:
#     savename = "%sFULL-PIC_Monthly_NHFLXEOF_TAUX_TAUY_centered%i.npz" % (rawpath,centered)
#     if correction:
#         savename = proc.addstrtoext(savename,correction_str)
#     ld = np.load(savename)
#     taux_pat_fin = ld['taux']
#     tauy_pat_fin = ld['tauy']
#     lon180 = ld['lon']
#     lat = ld['lat']
#     print("Loading wind-stress regression output from: %s" % savename)
#     print("Centering is: %s" %centered)
    

# # ------------------------------------------
# #%#%% Visualization for Wind Stress Regression
# # ------------------------------------------

# #% Individual monthly wind stress plots for EOF N
# im = 9
# N  = 1
# scaler   =    .75 # # of data units per arrow
# bboxplot = [-100,20,0,80]
# labeltau = 0.10
# slplevs  = np.arange(-400,500,100)
# flxlim   = [-30,30]
# mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')

# oint = 5
# aint = 5

# for im in tqdm(range(12)):
#     fig,ax = plt.subplots(1,1,figsize=(6,4),subplot_kw={'projection':ccrs.PlateCarree()})
#     ax = viz.add_coast_grid(ax,bbox=bboxplot)
    
#     pcm = ax.pcolormesh(lon180,lat,eofall[:,:,im,N].T,vmin=flxlim[0],vmax=flxlim[-1],cmap="RdBu_r")
#     cl  = ax.contour(lon180,lat,eofslp[:,:,im,N].T,levels=slplevs,colors='k',linewidth=0.75)
    
#     qv = ax.quiver(lon180[::oint],lat[::aint],
#                    taux_pat_fin[::oint,::aint,im,N].T,
#                    tauy_pat_fin[::oint,::aint,im,N].T,
#                    scale=scaler,color='gray',width=.008,
#                    headlength=5,headwidth=2,zorder=9)
#     ax.quiverkey(qv,1.10,1.045,labeltau,"%.2f $Nm^{-2}\sigma_{PC}^{-1}$" % (labeltau))
    
#     fig.colorbar(pcm,ax=ax,fraction=0.035)
#     ax.set_title("%s Wind Stress Associated with NHFLX EOF %i \n CESM-FULL"%(mons3[im],N+1))
#     savename = "%sCESM_FULL-PIC_WindStressMap_EOF%02i_month%02i.png" %(figpath,N+1,im+1)
#     plt.savefig(savename,dpi=150,bbox_inches='tight')

# #%% Seasonal NHFLX-SLP-Windstress plots

# season_idx  = [[11,0,1],[2,3,4],[5,6,7],[8,9,10]]
# season_name = ["DJF","MAM","JJA","SON"]
# scaler = 0.5
# fig,axs = plt.subplots(1,4,figsize=(16,3),subplot_kw={'projection':ccrs.PlateCarree()})
# for i in range(4):
    
#     sid   = season_idx[i]
#     sname = season_name[i]
#     ax    = axs.flatten()[i]
    
#     ax = viz.add_coast_grid(ax,bbox=bboxplot)
#     pcm = ax.pcolormesh(lon180,lat,eofall[:,:,sid,N].mean(2).T,vmin=flxlim[0],vmax=flxlim[-1],cmap="RdBu_r")
#     cl  = ax.contour(lon180,lat,eofslp[:,:,sid,N].mean(2).T,levels=slplevs,colors='k',linewidth=0.95)
    
#     qv = ax.quiver(lon180[::oint],lat[::aint],
#                taux_pat_fin[::oint,::aint,sid,N].mean(2).T,
#                tauy_pat_fin[::oint,::aint,sid,N].mean(2).T,
#                scale=scaler,color='gray',width=.008,
#                headlength=5,headwidth=2,zorder=9)
#     #ax.quiverkey(qv,1.10,1.045,labeltau,"%.2f $Nm^{-2}\sigma_{PC}^{-1}$" % (labeltau))
    
#     ax.set_title(sname)

# fig.colorbar(pcm,ax=axs.ravel().tolist(),orientation='vertical',shrink=0.35,pad=0.01)
# plt.suptitle("Seasonal Averages for EOF %i of $Q_{net}$ (colors), $SLP$ (contours), and Wind Stress (quivers)" % (N+1),y=0.94)
# savename = "%sCESM_FULL-PIC_WindStressMap_EOF%02i_seasonal.png" %(figpath,N+1)   
# plt.savefig(savename,dpi=150,bbox_inches='tight')

# #%% Test visualzation of the wind stress variable

# fig,ax = plt.subplots(1,1,figsize=(8,4),subplot_kw={'projection':ccrs.PlateCarree()})
# ax = viz.add_coast_grid(ax,bbox=[-180,180,-90,90])

# oint     = 7
# aint     = 7
# t        = 555
# labeltau = 0.1
# scaler   = 2

# #Contour the meridional wind
# pcm = ax.pcolormesh(lon360,lat,np.mean(fully,(0,1)),vmin=-.2,vmax=.2,cmap="RdBu_r")
# fig.colorbar(pcm,ax=ax,fraction = 0.025)
# qv  = ax.quiver(lon360[::oint],lat[::aint],
#                np.mean(fullx[:,:,::aint,::oint],(0,1)),
#                np.mean(fully[:,:,::aint,::oint],(0,1)),
#                scale=scaler,color='gray',width=.008,
#                headlength=5,headwidth=2,zorder=9)
# ax.quiverkey(qv,1.10,1.045,labeltau,"%.2f $Nm^{-2}\sigma_{PC}^{-1}$" % (labeltau))
# ax.quiverkey(qv,1.10,1.045,labeltau,"%.2f $Nm^{-2}\sigma_{PC}^{-1}$" % (labeltau))
# ax.set_title("Meridional Wind Stress (colors) and the wind stress vectors (arrows)")
# # End Result [lon x lat x mon x mode]

# # --------------------------------------
# #%% Part 3: Calculate the ekman velocity
# # --------------------------------------
# if calc_qek:
    
#     #% Load mixed layer depths
#     # ------------------------
#     st    = time.time()
#     dsmld = xr.open_dataset(rawpath+"HMXL_PIC.nc")
#     hmxl  = dsmld.HMXL.values # [lon180 x lat x time]
#     print("Load MLD in %.2fs"%(time.time()-st))

#     # Find the climatological mean
#     mld     = hmxl.reshape(288,192,int(hmxl.shape[2]/12),12)
#     mldclim = mld.mean(2)

#     # Convert cm --> meters
#     mldclim /= 100 
    
#     # First, lets deal with the coriolis parameter
#     # --------------------------------------------
#     f       = 2*omega*np.sin(np.radians(yy))
#     dividef = 1/f 
    
#     # Remove values around equator
#     dividef[np.abs(yy)<=6] = np.nan
#     if debug: # Test plot 1/f
#         fig,ax = plt.subplots(1,1,figsize=(6,4),subplot_kw={'projection':ccrs.PlateCarree()})
#         ax = viz.add_coast_grid(ax)
#         pcm = ax.pcolormesh(lon360,lat,dividef)
#         fig.colorbar(pcm,ax=ax)
    
#     # Remove coastal points 
#     xroll = msk * np.roll(msk,-1,axis=1) * np.roll(msk,1,axis=1)
#     yroll = msk * np.roll(msk,-1,axis=0) * np.roll(msk,1,axis=0)
#     mskcoastal = msk * xroll * yroll
    
#     # Scrap plot to examine values near the equator
#     #plt.pcolormesh(lon360,lat,dividef),plt.colorbar(),plt.ylim([-20,20])
#     _,mld360 = proc.lon180to360(lon180,mldclim)
#     mld360 = mld360.transpose(1,0,2) # lat x lon x time
    
#     # Calculate the anomalous ekman current
#     # -------------------------------------
#     u_ek = dividef[:,:,None,None] * tauy_pat  / (rho*mld360[:,:,:,None])
#     v_ek = dividef[:,:,None,None] * -taux_pat  / (rho*mld360[:,:,:,None])
    
#     # Transpose to from [mon x lat x lon] to [lat x lon x mon]
#     dSSTdx = dTdx.transpose(1,2,0)
#     dSSTdy = dTdy.transpose(1,2,0)
    
#     # Calculate ekman heat flux #[lat x lon x mon x N]
#     # ------------------------------------------------
#     #q_ek = cp0 * dividef[:,:,None,None] * (-tauy_pat*dSSTdx[:,:,:,None] + taux_pat*dSSTdy[:,:,:,None])
#     q_ek = -1* cp0 *(rho*mld360[:,:,:,None]) * (u_ek*dSSTdx[:,:,:,None] + v_ek*dSSTdy[:,:,:,None])
#     q_ek_msk = q_ek * mskcoastal[:,:,None,None] # Apply coastal mask
    
#     #% Save the output
#     # ----------------
#     invars  = [q_ek_msk,u_ek,v_ek]
#     outvars = []
#     for i in tqdm(range(len(invars))):
        
#         # Get variable
#         invar = invars[i]
        
#         # Change to lon x lat x otherdims
#         invar = invar.reshape(nlat,nlon,12*N_mode).transpose(1,0,2)
        
#         # Flip longitude
#         _,invar = proc.lon360to180(lon360,invar)
        
#         # Uncombine mon x N_mode
#         invar = invar.reshape(nlon,nlat,12,N_mode)
#         outvars.append(invar)
        
#     q_ek180,u_ek180,v_ek180 = outvars

#     # Save output output
#     savename = "%sFULL-PIC_Monthly_NHFLXEOF_Qek_centered%i.npz" % (rawpath,centered)
#     if correction:
#         savename = proc.addstrtoext(savename,correction_str)
#     np.savez(savename,**{
#         'q_ek':q_ek180,
#         'u_ek':u_ek180,
#         'v_ek':v_ek180,
#         'lat':lat,
#         'lon':lon180})
#     print("Saving Ekman Forcing to: %s" % savename)
# else:
    
#     savename = "%sFULL-PIC_Monthly_NHFLXEOF_Qek_centered%i.npz" % (rawpath,centered)
#     if correction:
#         savename = proc.addstrtoext(savename,correction_str)
#     ld = np.load(savename)
#     q_ek180 = ld['q_ek']
#     u_ek180 = ld['u_ek']
#     v_ek180 = ld['v_ek']
#     print("Loading Ekman Forcing from: %s" % savename)
#     print("Centering is: %s" %centered)


# #%% Visualizations for Q-ek calculations
# # --------------------------------------

# #% %est plot temperature 
# im = 0
# fig,ax = plt.subplots(1,1,figsize=(6,4),subplot_kw={'projection':ccrs.PlateCarree()})
# ax = viz.add_coast_grid(ax)
# pcm = ax.pcolormesh(lon360,lat,ts_monmean[im,:,:]*msk)
# fig.colorbar(pcm,ax=ax)

# #%% Visualize ekman advection


# # Option
# #im = 0 # Month Index (for debugging)
# N  = 0 # Mode Index
# viz_tau      = False # True: Include wind stress quivers
# contour_temp = True # True: contour SST ... False: contour q_ek


# # Silly things: flip back to degrees east just for plotting :(....
# _,q_ek_msk = proc.lon180to360(lon180,q_ek180)
# _,u_ek = proc.lon180to360(lon180,u_ek180)
# _,v_ek = proc.lon180to360(lon180,v_ek180)
# q_ek_msk = q_ek_msk.transpose(1,0,2,3)
# u_ek = u_ek.transpose(1,0,2,3)
# v_ek = v_ek.transpose(1,0,2,3)


# # U_ek quiver options
# scaler   = 0.1 
# labeltau = 0.01

# # Q_ek contour levels
# clevs =np.arange(-30,40,10)
# lablevs = np.arange(-30,35,5)

# # Temperature contour levels
# tlm = [275,310] 
# tlevs = np.arange(tlm[0],tlm[1]+1,1)
# tlab  = np.arange(tlm[0],tlm[1]+5,5)

# # Projection
# for im in range(12):
    
#     fig,ax = plt.subplots(1,1,figsize=(6,4),subplot_kw={'projection':ccrs.PlateCarree()})
#     ax = viz.add_coast_grid(ax,bbox=bboxplot)
#     pcm = ax.pcolormesh(lon360,lat,(q_ek_msk)[:,:,im,N],vmin=-25,vmax=25,cmap="RdBu_r")
    
#     if contour_temp:
#         cl = ax.contour(lon360,lat,ts_monmean[im,:,:]*msk,levels=tlevs,colors='k',linewidths=0.75)
#         ax.clabel(cl,levels=tlab)
#     else:
#         cl = ax.contour(lon360,lat,(q_ek_msk)[:,:,im,N],levels=clevs,colors='k',linewidths=0.75)
    
#     fig.colorbar(pcm,ax=ax,fraction=0.035)
    
#     qv = ax.quiver(lon360[::oint],lat[::aint],
#                    u_ek[::aint,::oint,im,N],
#                    v_ek[::aint,::oint,im,N],
#                    scale=scaler,color='gray',width=.008,
#                    headlength=5,headwidth=2,zorder=9)
#     ax.quiverkey(qv,1.1,1.035,labeltau,"%.3f $m/s$" % (labeltau))
#     if viz_tau:
#         qv2 = ax.quiver(lon180[::oint],lat[::aint],
#                    taux_pat_fin[::oint,::aint,im,N].T,
#                    tauy_pat_fin[::oint,::aint,im,N].T,
#                    scale=0.5,color='blue',width=.008,
#                    headlength=5,headwidth=2,zorder=9)
#     if contour_temp:
#         ax.set_title(r"%s $Q_{ek}$ (Contour Interval: 10 $\frac{W}{m^{2}}$; 1 $^{\circ}C$)" % (mons3[im]))
#     else:
        
#         ax.set_title(r"%s $Q_{ek}$ (Contour Interval: 10 $\frac{W}{m^{2}}$)" % (mons3[im]))
    
#     savename = "%sCESM_FULL-PIC_Qek-Map_EOF%02i_month%02i.png" %(figpath,N+1,im+1)
#     plt.savefig(savename,dpi=150,bbox_inches='tight')

# # -------------------------------------------------------------
# # %% PART 4: Save selected number (based on selected threshold)
# # -------------------------------------------------------------

# # First, obtain index for selected threshold for variance explained
# # Below section is from NHFLX_EOF_monthly, copied 2021.01.04

# # Calculate cumulative variance at each EOF
# cvarall = np.zeros(varexpall.shape)
# for i in range(N_mode):
#     cvarall[i,:] = varexpall[:i+1,:].sum(0)

# # Select threshold based on variance explained
# vthres  = 0.90
# thresid = np.argmax(cvarall>vthres,axis=0)
# thresperc = []
# for i in range(12):
    
#     print("Before")
#     print(cvarall[thresid[i]-1,i])
#     print("After")
#     print(cvarall[thresid[i],i])
    
#     # Append percentage
#     thresperc.append(cvarall[thresid[i],i])
# thresperc = np.array(thresperc)
# if debug:
#     fig,ax = plt.subplots(1,1,figsize=(5,4))
#     ax.bar(mons3,thresid,color=[0.56,0.90,0.70],alpha=0.80)
#     ax.set_title("Number of EOFs required \n to explain %i"%(vthres*100)+"% of the $Q_{net}$ variance")
#     #ax.set_yticks(ytk)
#     ax.set_ylabel("# EOFs")
#     ax.grid(True,ls='dotted')
    
#     rects = ax.patches
#     labels = thresid
    
#     for rect, label in zip(rects, labels):
#         height = rect.get_height()
#         ax.text(
#             rect.get_x() + rect.get_width() / 2, height + -5, label, ha="center", va="bottom"
#         )
        
# #%% Take threshold from variable (NOTE: no correction is applied!)
# eofcorr = 0

# qekforce = q_ek180.copy() # [lon x lat x month x pc]
# cvartest = cvarall.copy()
# for i in range(12):
#     # Set all points after crossing the variance threshold to zero
#     stop_id = thresid[i]
#     print("Variance of %f  at EOF %i for Month %i "% (cvarall[stop_id,i],stop_id+1,i+1))
#     qekforce[:,:,i,stop_id+1:] = 0
#     cvartest[stop_id+1:,i] = 0
# qekforce= qekforce.transpose(0,1,3,2) # [lon x lat x pc x mon]

# # Cut to maximum EOF
# nmax = thresid.max()
# qekforce = qekforce[:,:,:nmax+1,:]
# savenamefrc = "%sQek_eof_%03ipct_%s_eofcorr%i.npy" % (datpath,vthres*100,"FULL_PIC",eofcorr)
# if correction:
#     savenamefrc = proc.addstrtoext(savenamefrc,correction_str)
# np.save(savenamefrc,qekforce)

# print("Saved postprocessed Q-ek forcing to %s" % (savenamefrc))






# #%%% Old Scripts Below
# #%% Combine output with net heat flux

# q_ek180add = q_ek180.copy()
# q_ek180add[np.isnan(q_ek180)] = 0

# # Combine Heat Fluxes and save
# q_comb = eofall + q_ek180add


# #%% Save a selected # of EOFS
# mcname = "SLAB-PIC"
# N_mode_choose = 50
# eofforce      = q_comb.copy()
# eofforce      = eofforce.transpose(0,1,3,2) # lon x lat x pc x mon
# eofforce      = eofforce[:,:,:N_mode_choose,:]
# savenamefrc   = "%sflxeof_qek_%ieofs_%s.npy" % (rawpath,N_mode_choose,mcname)
# np.save(savenamefrc,eofforce)
# print("Saved data to "+savenamefrc)



# #%%

# # Calculate correction factor
# eofcorr  = True
# if eofcorr:
#     ampfactor = 1/thresperc
# else:
#     ampfactor = 1

# eofforce = q_comb.copy() # [lon x lat x month x pc]
# cvartest = cvarall.copy()
# for i in range(12):
#     # Set all points after crossing the variance threshold to zero
#     stop_id = thresid[i]
#     print("Variance of %f  at EOF %i for Month %i "% (cvarall[stop_id,i],stop_id+1,i+1))
#     eofforce[:,:,i,stop_id+1:] = 0
#     cvartest[stop_id+1:,i] = 0
# eofforce = eofforce.transpose(0,1,3,2) # [lon x lat x pc x mon]

# if eofcorr:
#     eofforce *= ampfactor[None,None,None,:]

# # Cut to maximum EOF
# nmax = thresid.max()
# eofforce = eofforce[:,:,:nmax+1,:]

# savenamefrc = "%sflxeof_q-ek_%03ipct_%s_eofcorr%i.npy" % (datpath,vthres*100,"SLAB-PIC",eofcorr)
# np.save(savenamefrc,eofforce)


# #%% Load data again (optional) and save just the EOFs for a given season

# loadagain       = True
# N_mode_choose   = 2
# mcname          = "SLAB-PIC"
# saveid          = [5,6,7] # Indices of months to average over
# savenamenew     = "%sflxeof_qek_%ieofs_%s_JJA.npy" % (rawpath,N_mode_choose,mcname)

# if loadagain:
#     savenamefrc   = "%sflxeof_qek_%ieofs_%s.npy" % (rawpath,N_mode_choose,mcname)
#     if correction:
#         savenamefrc = proc.addstrtoext(savenamefrc,correction_str)
#     eofforce = np.load(savenamefrc)

# eofforceseas = np.mean(eofforce[:,:,:,saveid],-1,keepdims=True) # Take mean along month axis
# eofforceseas = np.tile(eofforceseas,12) # Tile along last dimension
# np.save(savenamenew,eofforceseas)
# print("Saved data to "+savenamenew)


# # Check plot
# N = 0
# bboxplot = [-100,20,0,80]
# fig,axs = plt.subplots(3,4,figsize=(12,4),subplot_kw={'projection':ccrs.PlateCarree()})
# for im in range(12):
#     ax = axs.flatten()[im]
#     ax = viz.add_coast_grid(ax,bbox=bboxplot)
#     ax.pcolormesh(lon180,lat,eofforceseas[:,:,0,im].T,vmin=-30,vmax=30,cmap="RdBu_r")
#     ax.set_title("Mon %i"%(im+1))
    


# #%%
# plotvars  = [eofall,q_ek180add,q_comb]
# plotlabs  = ["$Q_{net}$ ($Wm^{-2}$)","$Q_{ek}$ ($Wm^{-2}$)","$Q_{total}$ ($Wm^{-2}$)"]

# N = 30


# for im in tqdm(range(12)):
#     fig,axs = plt.subplots(1,3,figsize=(12,4),subplot_kw={'projection':ccrs.PlateCarree()})
#     for i in range(3):
#         ax = axs.flatten()[i]
#         ax = viz.add_coast_grid(ax,bbox=bboxplot)
#         pcm = ax.pcolormesh(lon180,lat,plotvars[i][:,:,im,N].T,vmin=-5,vmax=5,cmap="RdBu_r")
#         fig.colorbar(pcm,ax=ax,fraction=0.035)
#         ax.set_title(plotlabs[i])
#     plt.suptitle("EOF %i (%s)" % (N+1,mons3[im] ))
    
#     savename = "%sCESM_FULL-PIC_AddQek_EOF%02i_month%02i.png" %(figpath,N+1,im+1)
#     plt.savefig(savename,dpi=150,bbox_tight='inches')



