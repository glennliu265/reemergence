#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Empirically estimate damping of detrained anomalies (Td', Sd')
at a single point in CESM1-LENs Historical.

Works with output from 
- repair_file_SALT_CESM1.py
- get-pt-data-stormtrack (legacy?)
- Includes pointwise figures from TCM march 2024


Tries a few different methods
(1): Depth Detrainment Damping
(2): Corr (Detrain, Entrain) @ Surface
(3): Corr (Detrain, Entrain) @ Depth

To Do:
- Rework to be compatible with extract_file_loop

Copied from Td Sd decay vertical on 2024.01.25
Created on Thu Jan 25 23:08:42 2024

"""
#%% ===========================================================================
#%% Setup

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
from tqdm import tqdm
import scipy as sp

# %% Import Custom Modules

amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/"  # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc, viz
import amv.loaders as dl
import scm

# %% Set data paths

# Select Point
lonf   = 330
latf   = 50
locfn, loctitle = proc.make_locstring(lonf, latf)

# Calculation Settings
lags   = np.arange(0,37,1)
lagmax = 3 # Number of lags to fit for exponential function 

# Indicate Paths
datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/lon%s_lat%s/" % (
    lonf, latf)
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20240322/"
proc.makedir(figpath)
outpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/%s/" % locfn

# Other toggles
debug = True # True to make debugging plots

# Plotting Stuff
mons3 = proc.get_monstr(nletters=3)

#%% Functions ---

def calc_acf_ens(tsens,lags):
    # tsens is the anomalized values [yr x mon x z]
    acfs_mon = []
    for im in range(12):
        basemonth   = im+1
        varin       = tsens[:,:,:]  # Month x Year x Npts
        out         = proc.calc_lagcovar_nd(varin, varin, lags, basemonth, 1)
        acfs_mon.append(out)
        # <End Month Loop>
    return np.array(acfs_mon) # [Mon Lag Depth]

def fit_exp_ens(acfs_mon,lagmax):
    # acfs_mon [month x lag x depth] : Monthly lagged ACFs
    
    _,nlags,nz = acfs_mon.shape
    tau_est = np.zeros((12, nz))
    acf_est = np.zeros((12, nlags, nz))
    for im in range(12):
        for zz in range(nz):
            acf_in = acfs_mon[im, :, zz] # Select Depth and Month
            
            outdict             = proc.expfit(acf_in, lags, lagmax=lagmax)
            tau_est[im, zz]     = outdict['tau_inv'].copy()
            acf_est[im, :, zz]  = outdict['acf_fit'].copy()
    return tau_est,acf_est

def monstacker(scycle):
    return np.hstack([scycle,scycle[:1]])



# --------------------------------------------------------------------
#%% Load CESM1 variables (see repair_file_SALT_CESM1.py)
# --------------------------------------------------------------------

# Load SALT or TEMP at a point ----------------------------------------------------------------
# Paths and Names
vname = "TEMP"
if vname == "TEMP":
    ncname = "CESM1_htr_TEMP_repaired.nc"
    outvar = "T"
else:
    ncname  = "CESM1_htr_SALT_repaired.nc"
    outvar = "S"
ncsalt  = outpath + ncname
ds_salt = xr.open_dataset(ncsalt)

# Load
z       = ds_salt.z_t.values  # /100 NOTE cm --> meter conversion done in repair code
times   = ds_salt.time.values
salt    = ds_salt[vname].values  # [Ens x Time x Depth ]
nens, ntime, nz = salt.shape

if "repaired" not in ncname:
    print("Repairing File")
    # Repair File if needed
    # Set depths to zero
    salt_sumtime = salt.sum(1)[0,:]
    idnanz       = np.where(np.isnan(salt_sumtime))[0][0]
    salt = salt[:,:,:idnanz]
    z    = z[:idnanz] / 100
    nz    = len(z)
    
    for t in range(len(times)):
        for e in range(42):
            if np.all(np.isnan(salt[e,t,:])):
                print("ALL is NaN at t=%i, e =%i" % (t,e))
                salt[:,t,:] = 0

# Get strings for time
timesstr = ["%04i-%02i" % (t.year, t.month) for t in times]

# Get Ensemble Numbers
ens     = np.arange(nens)+1

# Load HBLT ----------------------------------------------------------------
# Paths and Names
mldname = "HMXL"
if mldname == "HBLT":
    
    mldpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/CESM_proc/"
    mldnc   = "HBLT_FULL_HTR_lon-80to0_lat0to65_DTFalse.nc"
    
    # Load and select point
    dsh         = xr.open_dataset(mldpath+mldnc)
    hbltpt      = dsh.sel(lon=lonf-360, lat=latf,
                     method='nearest').load()  # [Ens x time x z_t]
    
    # Compute Mean Climatology [ens x mon]
    hclim       = hbltpt.groupby('time.month').mean('time').squeeze().HBLT.values/100  # Ens x month, convert cm --> m
    
    # Compute Detrainment month
    kprev, _    = scm.find_kprev(hclim.mean(1)) # Detrainment Months #[12,]
    hmax        = hclim.mean(1).max() # Maximum MLD of seasonal cycle # [1,]
elif mldname == "HMXL":
    mldpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/mld/"
    mldnc   = "CESM1_HTR_FULL_HMXL_NAtl.nc"
    
    # Load and select point
    dsh       = xr.open_dataset(mldpath+mldnc)
    hbltpt      = dsh.sel(lon=lonf-360, lat=latf,
                     method='nearest').load()  # [Ens x time x z_t]
    
    # Compute Mean Climatology [ens x mon]
    hclim       = hbltpt.h.values
    
    
    # Compute Detrainment month
    kprev, _    = scm.find_kprev(hclim.mean(-1)) # Detrainment Months #[12,]
    hmax        = hclim.max()#hclim.mean(1).max() # Maximum MLD of seasonal cycle # [1,]


#%% Preprocess timeseries (Deseason/Detrend)
# --------------------------------------------------------------------
# Note, there should be no NaN values, accomplished through the "repair" script.

# 2A. Compute the seasonal cycle and monthly anomaly
# Get Seasonal Cycle
scycle, tsmonyr = proc.calc_clim(salt, 1, returnts=True)  # [ens x yr x mon x z]

# Compute monthly anomaly
tsanom          = tsmonyr - scycle[:, None, :, :]

# 2B. Remove the ensemble average

tsanom_ensavg   = np.nanmean(tsanom, 0)

tsanom_dt       = tsanom - tsanom_ensavg[None, ...]

# Check detrending
if debug:
    iz = 0 # Depth
    e  = 0 # Ensemble
    fig, ax = plt.subplots(1, 1)
    
    ax.plot(tsanom[e, :, :, iz].flatten(), label="Raw", c='red')
    ax.plot(tsanom_dt[e, :, :, iz].flatten(),
            label="Detrended", c='k', ls='dashed')
    ax.plot(tsanom_ensavg[:, :, iz].flatten(), label="Ens. Avg.", c="mediumblue")
    ax.legend()
    ax.set_title("Detrended Value at Depth z=%im, Ens %i" % (z[iz], e+1))

#%% Load lbd_d calculated with [preproc_detrainment_damping]
# --------------------------------------------------------------------

# First Load a file computed with the other script
path2           = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/ocn_var_3d/"
nc2             = "CESM1_HTR_FULL_lbd_d_params_%s_detrendensmean_lagmax3_ens01_regridNN.nc" % vname
ds2             = xr.open_dataset(path2+nc2)

# Load ACF and select point
acfpt_script    = ds2.acf_mon.sel(lon=lonf-360,lat=latf,method='nearest').load()


#%% Plotting Stuff


# Lag Correlations
xtks = np.arange(0,37,1)


#%% ===========================================================================
#%% Calculations
# -------------------------------------------------
#%% Method 1 [M1] Expfit depth-dependent detrainment
# -------------------------------------------------

#%% 1.1 Do Calculations

# Get some additional dimensions
nlags       = len(lags)

# Preallocate
lbd_d_all   = np.zeros((nens,12))          # estimated detrainment damping
tau_est_all = np.zeros((nens,12,nz))       # acf-fit damping
acf_est_all = np.zeros((nens,12,nlags,nz)) # Fitted ACF
acf_mon_all = np.zeros((nens,12,nlags,nz)) # Actual ACF

# Loop by ens
for e in tqdm(range(nens)):
    
    # Select ensemble data
    tsens     = tsanom_dt[e,:,:,:]       # Anomalies [yr x mon x z]
    hclim_ens = hclim[:,e]               # MLD Cycle [mon]
    
    # 3. Compute ACF
    acfs_mon        = calc_acf_ens(tsens.transpose(1,0,2)  ,lags) # [mon x lag x depth]
    
    # 4. Compute Expfit
    tau_est,acf_est = fit_exp_ens(acfs_mon,lagmax) # [mon x depth], [mon x lags x depth]
    
    # 5. Compute Detrainment Damping
    kprev,_ = scm.find_kprev(hclim_ens)
    lbd_d   = scm.calc_tau_detrain(hclim_ens,kprev,z,tau_est,debug=False)
    
    # Save Output
    
    lbd_d_all[e,:]       = lbd_d.copy()
    tau_est_all[e,:,:]   = tau_est.copy()
    acf_est_all[e,:,:,:] = acf_est.copy()
    acf_mon_all[e,:,:,:] = acfs_mon.copy()
    
    # <End Ens Loop> ---

#%% 1.2 Save output (depth dependent entrainment)

savename = "%s%sd_damping_CESM1_HTR_FULL_%s_%s_%ilagfig_lags%02i.npz" % (outpath,outvar,locfn,mldname,lagmax,lags[-1])
savedict = {
    "lbd_d"     :np.abs(lbd_d_all), # Note these are negative
    "tau_est"   :np.abs(tau_est_all), 
    "acf_est"   :acf_est_all,
    "acf_mon"   :acf_mon_all,
    "tsanom"    :tsanom_dt,
    "lags"      :lags,
    "z_t"       :z,
    "time"      :timesstr,
    "lagmax"    :lagmax,
    "hblt"      :hclim,
    }
np.savez(savename,**savedict,allow_pickle=True)
# -------------------------------------------------
#%% Method 2/3 [M2,3] Surf./Depth Corr(Detrain, Entrain)
# -------------------------------------------------
"""
Method 2: Corr(Detrain, Entrain) at surface
Method 3: Corr(Detrain, Entrain) at depth
"""
# Indicate Settings
surface      = False           # Set to True for Method (2), False for Method (3)
imshift      = 0               # How much to shift the calculation month backwards
#dtshift      = 0               # How much to shift the detrainent month backwards....
interpcorr   = 1               # Set to True to interp values
usedtdepth   = True            # Set to True to use detrain depth (rather than depth @ detraining months)

#% Recompute detrainment damping based on the month of detrainment
ts_surface   = tsanom[:,:,:,0] # ens x yr x mon 
nyr          = tsanom.shape[1]
expf3        = lambda t,b: np.exp(b*t)         # No c and A


#%% 2.1 Do Calculations

# Preallocate
corr_byens   = np.zeros((nens,12))
tau_byens    = np.zeros((nens,12))
acffit_byens = np.zeros((nens,12,nlags)) 

for im in tqdm(range(12)):
    # if im != 6:
    #     continue
    
    # Get Index of the detraining month
    detrain_mon = kprev[im]
    dtid        = int(np.floor(detrain_mon) - 1) # Overestimate to compensate for MLD variability
    
    if detrain_mon == 0:
        continue # Skip months where detrainment is occuring
        
    
    entrain_mon = im+1
    # 3 Shift cases. 
    # (1) detrain month (<) precedes entrain month. use all data
    # (2) detrain month (>) follows entrain detrain month, apply lag to correct
    # (3) detrain month = entrain month (Deepest MLD month), apply lag to correct
    if detrain_mon < entrain_mon:
        shift = 0 # Correlation with anomalies the same year
    else:
        shift = 1 # Correlation with anomalies the following year
    if detrain_mon >= (entrain_mon):
        entrain_mon = entrain_mon+12
    
    x    = [detrain_mon,entrain_mon]
    xlag = [0,entrain_mon-detrain_mon]
    
    corr_allens = []
    for e in range(nens):
        
        # Apply Shift to variable and select the ensemble, month, and level
        if surface:
            
            x1       = ts_surface[e,:(nyr-shift),dtid] # Detrain Anoms {Year}
            x2       = ts_surface[e,shift:,im-imshift]         # Entrain Anoms {Year}
        
        else:
            
            hdetrain = hclim[dtid,e]
            idz      = proc.get_nearest(hdetrain,z)
            x1       = tsanom[e,:(nyr-shift),dtid,idz]
            if (im-imshift < 0) and (imshift != 0): # For cases where it goes back to the last year 
                x2       = tsanom[e,:(nyr-shift),im-imshift,idz]
            else:
                x2       = tsanom[e,shift:,im-imshift,idz]
        
        # Compute the correlation
        corr_ens             = np.corrcoef(x1,x2)[0,1]
        
        # Save the output
        if interpcorr:
            # Also compute for np.ceil estimate (just shift the detrain month)
            dtidceil = int(np.ceil(detrain_mon) - 1)
            if surface:
                x1       = ts_surface[e,:(nyr-shift),dtidceil] # Detrain Anoms {Year}
                x2       = ts_surface[e,shift:,im-imshift]         # Entrain Anoms {Year}
            else:
                
                hdetrain = hclim[dtidceil,e]
                idz      = proc.get_nearest(hdetrain,z)
                x1       = tsanom[e,:(nyr-shift),dtidceil,idz]
                if (im-imshift < 0) and (imshift != 0): # For cases where it goes back to the last year 
                    x2       = tsanom[e,:(nyr-shift),im-imshift,idz]
                else:
                    x2       = tsanom[e,shift:,im-imshift,idz]
            
            
            corr_ens1 = np.corrcoef(x1,x2)[0,1]
            if (dtidceil) == (im-imshift): # For first detraining month, when detrainment month is too close to shifted month
                corr_ens = corr_ens
            else:
                corr_ens = np.interp(kprev[im],[dtid+1,dtidceil+1],[corr_ens,corr_ens1],)
        
        corr_byens[e,im]     = corr_ens
        
        # Compute the exponential fit
        y                    = [1,corr_ens]
        fitout               = proc.expfit(np.array(y),np.array(xlag),1)
        
        tau_byens[e,im]      = fitout['tau_inv']
        acffit_byens[e,im,:] = expf3(lags,fitout['tau_inv'])
        
        # if debug:

        #     use_script = False # Set to True to use ACF computed from other scriptd
        #     # Get ACFs from above and below the level
        #     hdetrain = hclim[dtid,e]
        #     hid_0    = np.argmin(np.abs(z - hdetrain))
        #     if use_script:
        #         acf0     = acfpt_script.isel(mon=im,z_t=iz)#acf_mon_all[e,dtid,:,hid_0]
        #         acf1     = acfpt_script.isel(mon=im,z_t=iz+1)#acf_mon_all[e,dtid,:,hid_0+1]
        #         acf_surf = acfpt_script.isel(mon=im,z_t=0)
        #     else:
        #         acf0     = acf_mon_all[e,dtid,:,hid_0]
        #         acf1     = acf_mon_all[e,dtid,:,hid_0+1]
        #         acf_surf = acf_mon_all[e,im,:,0]
            
            
            
        #     fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(8,4))
        #     title = "%s ACF Fit (Entrain Mon: %02i, Detrain Mon: %02i, Ens %02i)" % (vname,im+1,detrain_mon,e+1)
        #     ax,ax2  = viz.init_acplot(dtid,xtks,lags,title=title,ax=ax)
        #     ax.plot(lags,expf3(lags,fitout['tau_inv']),label=r"Exp Fit $\tau$=%.2f Months" % (1/np.abs(fitout['tau_inv'])))
        #     #ax.plot(lags,acf_surf,label="ACF (lag0=%02i, z=%.2f)" % (im+1,z[0]),color="orange")
            
            
        #     #ax.plot(lags,acf0,label="ACF (lag0=%02i, z=%.2f)" % (dtid+1,z[hid_0]),color="darkblue")
        #     #ax.plot(lags,acf1,label="ACF (lag0=%02i, z=%.2f)" % (dtid+1,z[hid_0+1]),color="magenta")
            
        #     ax.scatter(xlag,y,ls="None",marker="d",c='k',label="Detrain Corr: %.2f" % (corr_ens))
        #     ax.legend()
        #     ax.set_xticks(xtks)
        #     ax.set_xlim([0,24])
        #     ax2.set_xlim([0,24])
            
        #     savename = "%sSurface_Entrain_Estimates/EntrainMon%02i_Ens%02i" % (figpath,im+1,e+1)
        #     plt.savefig(savename,dpi=150,bbox_inches='tight')
        # #     plt.close()

#%% 2.2. Save Surface Estimates and Correlation
# For use in [stochmod_point_test_lbdd.py]

tau_byens   = np.abs(tau_byens)
coords      = dict(ens=np.arange(1,43,1),mon=np.arange(1,13,1))
da_tau2     = xr.DataArray(tau_byens,coords=coords,dims=coords,name='lbd_d')
da_corr2    = xr.DataArray(corr_byens,coords=coords,dims=coords,name='corr_d')
da_out      = xr.merge([da_tau2,da_corr2])
edict       = proc.make_encoding_dict(da_out)
savenametau = "%sLbdd_estimate_surface%0i_imshift%0i_interpcorr%i_%s.nc" % (outpath,surface,imshift,interpcorr,vname)
da_out.to_netcdf(savenametau)

# Save Deep Estimates
tau_3d      = np.abs(lbd_d_all)
da_tau3d    = xr.DataArray(tau_3d,coords=coords,dims=coords,name='lbd_d')
savenametau = "%sLbdd_estimate_deep_%s.nc" % (outpath,vname)
da_tau3d.to_netcdf(savenametau)


# -------------------------------------------------------------------
#%% Part 4: Testing a few different estimate styles (for correlation)
# -------------------------------------------------------------------

im = 10
e  = 0
# Copied section below from Part 2-3

# -------------------
#% Recompute detrainment damping based on the month of detrainment
ts_surface   = tsanom[:,:,:,0] # ens x yr x mon 
nyr          = tsanom.shape[1]
#surface      = False # Set to True for Method (2), False for Method (3)

expf3        = lambda t,b: np.exp(b*t)         # No c and A
corr_byens   = np.zeros((nens,12))
tau_byens    = np.zeros((nens,12))
acffit_byens = np.zeros((nens,12,nlags)) 

#lagxall = np.zeros((2,nens,12))

    
# Get Index of the detraining month
detrain_mon = kprev[im]
dtid        = int(np.floor(detrain_mon) - 1) # Overestimate to compensate for MLD variability

if detrain_mon == 0:
    print("Warning, doing calculations for a detraining month")

# entrain_mon = im+1
# # 3 Shift cases. 
# # (1) detrain month (<) precedes entrain month. use all data
# # (2) detrain month (>) follows entrain detrain month, apply lag to correct
# # (3) detrain month = entrain month (Deepest MLD month), apply lag to correct
# if detrain_mon < entrain_mon:
#     shift = 0 # Correlation with anomalies the same year
# else:
#     shift = 1 # Correlation with anomalies the following year
# if detrain_mon >= (entrain_mon):
#     entrain_mon = entrain_mon+12

# x    = [detrain_mon,entrain_mon]
# xlag = [0,entrain_mon-detrain_mon]

#%% Try different Estimation Methods

im_in            = 11

corr_shift = []
for ii in range(2):
    if ii == 0:
        imshift = 0
    else:
        imshift = -1
    
    corr_bymethod = []
    
    #% Method 1 (Round Down)
    dtfloor  = int(np.floor(kprev[im]-1))
    hdetrain = hclim[dtfloor,e]
    idz      = proc.get_nearest(hdetrain,z)
    x1       = tsanom[e,:(nyr-shift),dtfloor,idz]
    x2       = tsanom[e,shift:,im+imshift,idz]
    corr_ens = np.corrcoef(x1,x2)[0,1]
    print("Method 1: Rounding Down Corr(%s,%s)" % (mons3[dtfloor],mons3[im+imshift]))
    print("\tMonths : [kprev: %.2f, Detrain: %i, Entrain: %i]" % (kprev[im],dtfloor+1,x[1]))
    print("\tDepth  : %.2f" % (hdetrain))
    print("\tCorr   : %f" % (corr_ens))
    corr_bymethod.append(corr_ens)
    
    # Method 2 (Round Up)
    dtceil     = int(np.ceil(kprev[im]-1))
    hdetrain = hclim[dtceil,e]
    idz      = proc.get_nearest(hdetrain,z)
    x1       = tsanom[e,:(nyr-shift),dtceil,idz]
    x2       = tsanom[e,shift:,im+imshift,idz]
    corr_ens = np.corrcoef(x1,x2)[0,1]
    print("Method 2: Rounding Down Corr(%s,%s)" % (mons3[dtceil],mons3[im+imshift]))
    print("\tMonths : [kprev: %.2f, Detrain: %i, Entrain: %i]" % (kprev[im],dtceil+1,x[1]))
    print("\tDepth  : %.2f" % (hdetrain))
    print("\tCorr   : %f" % (corr_ens))
    corr_bymethod.append(corr_ens)
    
    # Method 3 (Interpolate Final Result)
    corr_interp = np.interp(kprev[im],[dtfloor+1,dtceil+1],[corr_bymethod[0],corr_bymethod[1]])
    print("Method 3: Interp Result...")
    print("\t%i (%f), %f (%f), %i (%f)" % (dtfloor+1,corr_bymethod[0],
                                           kprev[im],corr_interp,
                                           dtceil+1,corr_bymethod[1],
                                           ))
    corr_bymethod.append(corr_interp)
    
    corr_shift.append(corr_bymethod)
corr_shift = np.array(corr_shift)

#%% Recompute correlation, tau, and expfit for all ens

# seems kinda redundant. consider deleting this
# corr_allens = []
# for e in range(nens):
    
#     # Apply Shift to variable and select the ensemble, month, and level
#     if surface:
#         x1       = ts_surface[e,:(nyr-shift),dtid] # Detrain Anoms {Year}
#         x2       = ts_surface[e,shift:,im]         # Entrain Anoms {Year}
#     else:
#         hdetrain = hclim[dtid,e]
#         idz      = proc.get_nearest(hdetrain,z)
#         x1       = tsanom[e,:(nyr-shift),dtid,idz]
#         x2       = tsanom[e,shift:,im,idz]
    
#     # Compute the correlation
#     corr_ens             = np.corrcoef(x1,x2)[0,1]
#     corr_byens[e,im]     = corr_ens
    
#     # Compute the exponential fit
#     y                    = [1,corr_ens]
#     fitout               = proc.expfit(np.array(y),np.array(xlag),1)
    
#     tau_byens[e,im]      = fitout['tau_inv']
#     acffit_byens[e,im,:] = expf3(lags,fitout['tau_inv'])
# -------------------

#%% ===========================================================================
#%% Visualizations


#%% M1.1 Load m1 output
# <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>

# Load npz output (Note,)
savename = "%s%sd_damping_CESM1_HTR_FULL_%s_%s_%ilagfig_lags%02i.npz" % (outpath,outvar,locfn,mldname,lagmax,lags[-1])
ld       = np.load(savename,allow_pickle=True)

# Load some variables
lbd_d   = ld['lbd_d']
tau_est = ld['tau_est']
hclim   = ld['hblt'] # 'hblt' is the selected mld variable, indicated by mldname

#%% M1.2 Timescale vs MLD
# <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>

fig,axs = viz.init_monplot(2,1,figsize=(8,6))

ax      = axs[0]
for e in range(nens):
    plotvar = lbd_d[e,:]
    if e == 0:
        lab="Indv. Member"
    else:
        lab=""
    ax.plot(mons3,plotvar,label=lab,color="gray",alpha=0.25)
ax.plot(mons3,lbd_d.mean(0),label="Ens. Avg.",c="k")
ax.set_title("Estimated $\lambda^d$ (Salinity) @ %s" % (loctitle)) 
ax.set_ylabel("e-folding timescale ($month^{-1}$)")
ax.legend()

# Plot Kprev
ax = axs[1]
for e in range(nens):
    plotvar = hclim[:,e]
    if e == 0:
        lab="Indv. Member"
    else:
        lab=""
    ax.plot(np.arange(0,12,1),plotvar,label=lab,color="gray",alpha=0.25,zorder=-1)
    
hmu     = hclim.mean(1)
ax.set_xlim([0,11])
ax.plot(mons3,hclim.mean(1),label="Ens. Avg.",c="k")

ax.set_ylabel("MLD (meters)")

savename = "%sSd_damping_AllEns_%s_%ilagfig_lags%02i.png" % (figpath,locfn,lagmax,lags[-1])
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% M1.3.1 Visualize calculation steps (timeseries @ 1 ens and month)

# First, plot timeseries at a given timepoint


# # Select ensemble data
# 
# hclim_ens       = hclim[:,e]               # MLD Cycle [mon]

# # 3. Compute ACF
# acfs_mon        = calc_acf_ens(tsens,lags) # [mon x lag x depth]

# # 4. Compute Expfit
# tau_est,acf_est = fit_exp_ens(acfs_mon,lagmax) # [mon x depth], [mon x lags x depth]

# # 5. Compute Detrainment Damping
# kprev,_         = scm.find_kprev(hclim_ens)
# lbd_d           = scm.calc_tau_detrain(hclim_ens,kprev,z,tau_est,debug=False)

#% Plot the Timeseries

# Select Ensemble Member and target month
e         = 0
im        = 10

tsens     = tsanom_dt[e,:,:,:]       # Anomalies [yr x mon x z]
hmon      = hclim[im,:].mean(-1)
iz        = np.argmin(np.abs(z-hmon))
dtid      = int(kprev[im]-1)

fig,ax  = plt.subplots(1,1,constrained_layout=True,figsize=(10,3.5))
ax      = viz.add_ticks(ax)
plotvar = tsens[:,:,iz].flatten()
ax.plot(plotvar,color='orange',lw=1.25)
ax.set_title("SST @ z=%.2f [meters]" % (z[iz]))
ax.set_xlim([0,len(plotvar)])
ax.set_xlabel("Time [months]")
ax.set_ylabel("Temperature [$\degree C$]")
savename = "%sLbdd_Demo_%s_Timeseries_z%03i.png" % (figpath,vname,z[iz])
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)
ax.vlines(np.arange(dtid,len(tsens[:,:,iz].flatten()),12),ymin=-1,ymax=1,linewidths=1,zorder=-1) # Visualize lines at detrainment month

#%% M1.3.2 Plot the ACF and Exp Fit @ all depths

for iz in tqdm(range(len(z))):
    im = dtid
    
    lagmaxviz = 36
    xtks      = np.arange(0,37,2)
    fig,ax    = plt.subplots(1,1,constrained_layout=True,figsize=(10,3.5))
    
    ax,ax2= viz.init_acplot(im,xtks,lags,ax=ax,title="")
    
    for e in range(nens):
        ax.plot(lags,acf_mon_all[e,im,:,iz],label="", lw=1, marker="o", c='orange',alpha=0.15,zorder=-1)
        ax.plot(lags,acf_est_all[e,im,:,iz],label="",lw=1,marker="d",ls='dashed',c='gray',alpha=0.15,zorder=-1)
    
    mu    = acf_mon_all[:,im,:,iz].mean(0)
    sigma = acf_mon_all[:,im,:,iz].std(0)
    ax.plot(lags,mu,label="Ens. Mean ACF", lw=2.5, marker="o", c='orange')
    #ax.fill_between(lags,mu-sigma,mu+sigma,alpha=.2,color='orange')
    
    ax.set_xlim([0,lagmaxviz])
    ax2.set_xlim([0,lagmaxviz])
    
    ax.set_title("%s ACF (Lag 0 = %s, Depth: %.2f [meters])" % (vname,mons3[im],z[iz]))
    ax.axhline([0],ls='solid',color="k",lw=.75)
    ax.axhline([1/np.exp(1)],ls='dashed',color="k",label="1/e",lw=.75)
    
    ax.plot(lags,acf_est_all[:,im,:,iz].mean(0),lw=2.5,marker="d",ls='dashed',c='gray',
            label=r"Ens. Mean ACF fit, $\tau$=%.3f months" % ((-1/tau_est_all[:,im,iz]).mean(0)))
    
    ax.legend()
    savename = "%sLbdd_Demo_%s_ACF_z%03i.png" % (figpath,vname,z[iz])
    plt.savefig(savename,dpi=150,transparent=True,bbox_inches='tight')
    plt.close()

#%% M1.3.3 Check the estimation for a month/ensemble

# 1) Indicate Ensemble and Base Month (Entrain Month)
im        = 1
e         = 0

# 2) Locate the detrainment month 
dtmon     = kprev[im]
dtid      = int(dtmon-1)

# 3) Identify corresponding depth
hdetrain  = hclim[dtid,e]
iz        = np.argmin(np.abs(z-hdetrain))

# 4) Select temperature timeseries at depth of detrainment month
ts_sel    = tsanom_dt[e,:,:,iz]       # Anomalies [yr x mon]

# <0> Plot the timeseries
fig,ax  = plt.subplots(1,1,constrained_layout=True,figsize=(10,3.5))
ax      = viz.add_ticks(ax)
ax.set_title("%s @ z=%.2f [meters]" % (vname,z[iz]))
ax.plot(ts_sel.flatten(),color='orange',lw=1.25)
#ax.plot(tsanom_dt[:,:,:,iz].mean(0).flatten(),label="Ens Mean") # Plot Ens Mean To Check
ax.set_xlim([0,len(plotvar)])
ax.set_xlabel("Time [months]")
ax.set_ylabel("Temperature [$\degree C$]")
ax.legend()

# Check deseason and detrend
#plt.plot(mons3,tsanom_dt[e,:,:,iz].mean(0)) # Check Seasonal Cycle

# 5) Compute ACF from detraining month
basemonth = dtid+1
acf   = proc.calc_lagcovar(ts_sel.T,ts_sel.T,lags,basemonth,0)
acfdt = proc.calc_lagcovar(ts_sel.T,ts_sel.T,lags,basemonth,1)

acfscript = ds2.acf_mon.isel(z_t=iz,mon=im).sel(lat=latf,lon=lonf-360,method='nearest').load()

# Visualize it
fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12,4.5))
title = "%s ACF Fit (Entrain Mon: %02i, Detrain Mon: %02i, Ens %02i)" % (vname,im+1,basemonth,e+1)
ax,ax2  = viz.init_acplot(im,xtks,lags,title=title,ax=ax)

ax.plot(lags,acf,label="ACF Calc")
ax.plot(lags,acfdt,label="ACF Calc Detrend")
ax.plot(lags,acf_mon_all[e,dtid,:,iz],label="ACF Func")
ax.plot(lags,acfscript,label="ACF Script")
ax.legend()

#%% M1.4 Plot Timescale Est. (tau, depth vs month contours)

ylim       = [0,500]
vlm        = [0,50]
pmesh      = False
cints      = np.arange(0,63,3)

# Timescales
tau_script = ds2.tau.sel(lon=lonf-360,lat=latf,method='nearest').T #  (mon: 12, z_t: 44, lat: 48, lon: 65)
plotvar    = 1/np.abs(tau_script)


fig,ax     = plt.subplots(1,1,constrained_layout=True,figsize=(12,5))

if pmesh:
    pcm = ax.pcolormesh(ds2.mon,ds2.z_t,plotvar,cmap='cmo.ice_r',vmin=vlm[0],vmax=vlm[-1])
else:
    pcm = ax.contourf(ds2.mon,ds2.z_t,plotvar,cmap='cmo.ice_r',levels=cints,extend='both')
    cl  = ax.contour(ds2.mon,ds2.z_t,plotvar,colors='dimgray',levels=cints,linewidths=0.75)
    ax.clabel(cl)

ax.plot(np.arange(1,13,1),hclim.mean(-1),color="k")
ax.set_ylim(ylim)
ax.invert_yaxis()

ax.set_xticks(np.arange(1,13,1),mons3)

ax.scatter((kprev)[kprev!=0],hclim.mean(1)[kprev!=0.],marker="x",c='red')

cb = fig.colorbar(pcm,ax=ax,pad=0.05,orientation='horizontal',fraction=0.03)
cb.set_label("E-folding Timescale (Months)")

ax.set_title("Estimated E-folding Timescale for %s" % vname )

savename = "%sLbdd_EstTau_DepthvMonth_%s.png" % (figpath,vname)
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)

# -----------------------------------------------------------------------------
#%% M2.1 Plot lbd_d Estimates
# Surface vs. 3D Detrainment Estimates (Scatter, Ens Mean, and 1 Stdev)
# 1/Month(Damping Values)
fig,axs = plt.subplots(2,1,constrained_layout=True,figsize=(12,8))

ax = axs[0]
for e in range(nens):
    
    ax.plot(mons3,lbd_d_all[e,:],lw=0.75,alpha=.75,marker="x",ls='none')

mu    = lbd_d_all[:,:].mean(0)
sigma = lbd_d_all[:,:].std(0)
ax.plot(mons3,mu,lw=0.75,alpha=1,marker="d",ls='solid',c="k")
ax.fill_between(mons3,mu-sigma,mu+sigma,alpha=0.2,color='gray')

ax.set_title("Estimated Deep/Detrainment Damping")
ax.set_ylabel("Damping (1/month)")
ax.set_xlabel("Entraining Month")

ax = axs[1]
for e in range(nens):
    
    ax.plot(mons3,tau_byens[e,:],lw=0.75,alpha=.75,marker="x",ls='none')

mu    = tau_byens[:,:].mean(0)
sigma = tau_byens[:,:].std(0)
ax.plot(mons3,mu,lw=0.75,alpha=1,marker="d",ls='solid',c="k")
ax.fill_between(mons3,mu-sigma,mu+sigma,alpha=0.2,color='gray')

ax.set_title("Estimated Deep/Detrainment Damping")
ax.set_ylabel("Damping (1/month)")
ax.set_xlabel("Entraining Month")

#ax.set_ylim([1,-1])

savename = "%sLbdd_Demo_EnsSpread_%s_ACF.png" % (figpath,vname)
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)


#%% M2.2 Plot tau estimates
# Plot Surface vs. 3D Detrainment Estimates (Scatter, Ens Mean, and 1 Stdev)
# Timescale (in months)

fig,axs = plt.subplots(2,1,constrained_layout=True,figsize=(12,8))

ax = axs[0]

lbd3d   = np.abs(1/lbd_d_all) # Fitted Lbd (from depths)
lbdsurf = np.abs(1/tau_byens) # Fitted Lbd (surface values)

for e in range(nens):
    
    ax.plot(mons3,lbd3d[e,:],lw=0.75,alpha=.75,marker="x",ls='none')
 
mu    = np.nanmean(lbd3d[:,:],0)
sigma = np.nanstd(lbd3d[:,:],0)
ax.plot(mons3,mu,lw=0.75,alpha=1,marker="d",ls='solid',c="k")
ax.fill_between(mons3,mu-sigma,mu+sigma,alpha=0.2,color='gray')

ax.set_title("Estimated Deep/Detrainment Damping (Depth-Varying Estimate)")
ax.set_ylabel("Damping Timescale (months)")
ax.set_xlabel("Entraining Month")

ax = axs[1]
for e in range(nens):
    
    ax.plot(mons3,lbdsurf[e,:],lw=0.75,alpha=.75,marker="x",ls='none')

mu    = lbdsurf[:,:].mean(0)
sigma = lbdsurf[:,:].std(0)
ax.plot(mons3,mu,lw=0.75,alpha=1,marker="d",ls='solid',c="k")
ax.fill_between(mons3,mu-sigma,mu+sigma,alpha=0.2,color='gray')

ax.set_title("Surface Entrain-Month-Based Estimate")
ax.set_ylabel("Damping Timescale (months)")
ax.set_xlabel("Entraining Month")

savename = "%sLbdd_Demo_EnsSpread_%s_ACF_Timescale.png" % (figpath,vname)
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)

#%% M2.3 Plot correlations and detrain month


fig,axs = plt.subplots(2,1,constrained_layout=True,figsize=(12,8))

mons3stack = monstacker(mons3)
plotx      = np.arange(1,14)

ax = axs[0]
ax = viz.viz_kprev(hclim.mean(1),kprev,ax=ax,lw=3)
ax.set_xticklabels(mons3stack)
ax = viz.add_ticks(ax)
ax.set_title("MLD Cycle and Detrainment Months")
for im in range(12):
    if kprev[im] == 0:
        ax.axvline([im+1],ls='solid',c='gray',lw=0.75)
    
# plot Correlation
ax = axs[1]
for e in range(nens):
    plotvar  = monstacker(corr_byens[e,:])
    ax.scatter(plotx,plotvar,marker="x")
# Plot mean.stdv
mu    = monstacker(corr_byens.mean(0))
sigma = monstacker(corr_byens.std(0))

ax.plot(plotx,mu,lw=3,alpha=1,marker="d",ls='solid',c="k",label="Ens. Mean")
ax.fill_between(plotx,mu-sigma,mu+sigma,alpha=0.2,color='gray',label="1$\sigma(Ens.)$")

ax.set_title("Corr(Detraining SST,Entraining SST)")

ax.set_ylabel("Correlation")
ax.set_xlim([1,13])

for im in range(12):
    if im == 0:
        lbl="No Entrainment"
    else:
        lbl=""
        
    if kprev[im] == 0:
        ax.axvline([im+1],ls='solid',c='gray',lw=0.75,label=lbl)
ax = viz.add_ticks(ax) 
ax.set_xticks(plotx,labels=mons3stack) 
ax.legend() 
    
rhocrit = proc.ttest_rho(0.05,1,86)
ax.axhline([rhocrit],lw=0.75,c="r",ls='dotted')

savename = "%sCorrelation_Surface_%s_ACF_Timescale.png" % (figpath,vname)
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)
    
# ------------------------------------------------------------
#%% M3.3 Load and compare correlations at the surface and at depth
# ------------------------------------------------------------

ds_taus = []
for surface in range(2):
    savenametau = "%sLbdd_estimate_surface%0i_%s.nc" % (outpath,surface,vname)
    ds2         = xr.open_dataset(savenametau)
    ds_taus.append(ds2.load())



snames = ["Detrainment Depth","Surface"]
scolors = ["royalblue","deeppink"]


# Make the Detrainment Strings (Just stole it from above)
xstr_mons = []
for im in tqdm(range(12)):
    
    # Get Index of the detraining month
    detrain_mon = kprev[im]
    dtid        = int(np.floor(detrain_mon) - 1) # Overestimate to compensate for MLD variability
    
    
    entrain_mon = im + 1 # FOR VISUALIZATION ADD 1
    
    if detrain_mon >= (entrain_mon):
        entrain_mon =  entrain_mon + 12
    if detrain_mon == 0:
        xstr = "%s\n"  % mons3[im]
    else:
        xstr = "%s\n"  % mons3[im] + r"$\rho$(%i,%i)" % (detrain_mon,entrain_mon)
    xstr_mons.append(xstr)
    x    = [detrain_mon,entrain_mon]
    
#%% Compute decay factor from damping timescales

lbd_d_script = ds2.lbd_d#.sel(lon=lonf-360,lat=latf,method='nearest').load() * -1


# decayfacs_all = []
# for dd in range(2):
    
#     decayfacs_taumon = []
    
#     for im in range(12):
        
#         m     = im+1
#         invar = lbd_d_script#ds_taus[dd].lbd_d.mean('ens')
#         dfacs = scm.calc_Td_decay_factor(kprev,m,invar)
#         dfacs = np.where(dfacs==1.,0,dfacs)
#         decayfacs_taumon.append(dfacs)
#     decayfacs_all.append(decayfacs_taumon)

# decayfacs_all = np.array(decayfacs_all)
delta_t = np.arange(1,13) - kprev
for tt in range(12):
    if delta_t[tt] <= 0: # For deepest layer and year crossings, add 12 months
        delta_t[tt] += 12
delta_t[kprev==0.] = np.nan
corr_timescales    = np.exp(-lbd_d_script * delta_t[None,:])
# np.exp(lbd_d_script * delta_t)


# <o> -- <o> -- <o> -- <o> -- <o> -- <o>
#%% Compare Deep and Surface Estimates for Correlation
# <o> -- <o> -- <o> -- <o> -- <o> -- <o>

xtks = np.arange(0,12,1)

fig,ax=viz.init_monplot(1,1,figsize=(10,4.5))

for ss in range(2):
    
    invar = ds_taus[ss].corr_d
    
    for e in range(42):
        
        plotvar = invar.isel(ens=e)
        ax.plot(mons3,plotvar,color=scolors[ss],alpha=0.1,label="")
        
    
    mu     = np.nanmean(invar,0)
    sigma  = np.nanstd(invar,0)
    
    ax.plot(mons3,mu,color=scolors[ss],alpha=1,label=snames[ss],marker="d",zorder=5)
    ax.fill_between(mons3,mu-sigma,mu+sigma,color=scolors[ss],alpha=0.3,label="")

# Plot correlation for expfit
for e in range(42):
    ax.plot(mons3,corr_timescales[e,:],color="k",alpha=0.1,label="")
mu     = np.nanmean(corr_timescales,0)
sigma  = np.nanstd(corr_timescales,0)
ax.plot(mons3,mu,color="k",alpha=1,label="Exp. Fit Estimate",marker="o",zorder=5,ls='dashed',lw=0.8)
ax.fill_between(mons3,mu-sigma,mu+sigma,color="k",alpha=0.3,label="")

ax.legend()
# for ss in range(1):
#     mu = decayfacs_all[ss]
#     ax.plot(mons3,mu,color="midnightblue",alpha=1,label=snames[ss] + "Exp(-Tau)",marker="d",ls='dashed',lw=2.5,)

ax.set_title("Corr(Detrain,Entrain) for %s" % (vname))
ax.set_xticks(xtks,labels=xstr_mons)
ax.set_ylabel("Correlation")
savename = "%sCorrelation_Surface_v_Vary_%s.png" % (figpath,vname)
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)


# <o> -- <o> -- <o> -- <o> -- <o> -- <o> --
#%% Compare Expfit and Correlation-Based Approach
# <o> -- <o> -- <o> -- <o> -- <o> -- <o> --

xtks = np.arange(0,12,1)

fig,ax=viz.init_monplot(1,1,figsize=(10,4.5))

ss=0
    
invar = da_out.corr_d

for e in range(42):
    
    plotvar = invar.isel(ens=e)
    ax.plot(mons3,plotvar,color=scolors[ss],alpha=0.1,label="")
    

mu     = np.nanmean(invar,0)
sigma  = np.nanstd(invar,0)

ax.plot(mons3,mu,color=scolors[ss],alpha=1,label="Correlation-Based",marker="d",zorder=5)
ax.fill_between(mons3,mu-sigma,mu+sigma,color=scolors[ss],alpha=0.3,label="")



# Plot correlation for expfit
for e in range(42):
    ax.plot(mons3,corr_timescales[e,:],color="k",alpha=0.1,label="")
mu     = np.nanmean(corr_timescales,0)
sigma  = np.nanstd(corr_timescales,0)
ax.plot(mons3,mu,color="k",alpha=1,label="Exp. Fit Estimate",marker="o",zorder=5,ls='dashed',lw=0.8)
ax.fill_between(mons3,mu-sigma,mu+sigma,color="k",alpha=0.3,label="")


ax.legend()
# for ss in range(1):
#     mu = decayfacs_all[ss]
#     ax.plot(mons3,mu,color="midnightblue",alpha=1,label=snames[ss] + "Exp(-Tau)",marker="d",ls='dashed',lw=2.5,)


ax.set_title("Corr(Detrain,Entrain) for %s" % (vname))
ax.set_xticks(xtks,labels=xstr_mons)
ax.set_ylabel("Correlation")
savename = "%sCorrelation_Expfit_v_Corr_%s.png" % (figpath,vname)
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)


#%% Plot Output using the different results

fns = ["Lbdd_estimate_surface0_imshift0_interpcorr0_%s.nc",
       "Lbdd_estimate_surface0_imshift0_interpcorr1_%s.nc",
       "Lbdd_estimate_surface0_imshift1_interpcorr0_%s.nc",
       "Lbdd_estimate_surface0_imshift1_interpcorr1_%s.nc"]

fns = [f % vname for f in fns]

expnames = ["No shift, no interp",
            "No shift, interp",
            "Shift, no interp",
            "Shift, interp"]

ecols = ["royalblue",
         "hotpink",
         "midnightblue",
         "orange"]

els    = ["solid",
          "dashed",
          "solid",
          "dashed"]

nexps       = len(fns)
ds_all      = []
for ff in range(nexps):
    print("Loading %s " % fns[ff])
    ds = xr.open_dataset(outpath+fns[ff]).corr_d.load()
    ds_all.append(ds)

#%% Plot each one

fig,ax = viz.init_monplot(1,1,figsize=(12,4.5))

mus = []

for ff in range(nexps):
    
    for e in range(nens):
        
        plotvar = ds_all[ff].isel(ens=e)
        ax.plot(mons3,plotvar,alpha=0.1,c=ecols[ff],zorder=-2)
    mu    = ds_all[ff].mean('ens')
    sigma = ds_all[ff].std('ens')
    mus.append(mu)
    ax.plot(mons3,mu,color=ecols[ff],label="%s"% (expnames[ff]),lw=2.5,ls=els[ff],zorder=9)
    ax.fill_between(mons3,mu-sigma,mu+sigma,alpha=.2,color=ecols[ff],zorder=4)
ax.legend(fontsize=14,ncol=2)
   
ax.set_title("Interpolation and Correlation Tests (%s)" % vname,fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_ylabel("Correlation(Detrain,Entrain)",fontsize=18)

savename = "%sAblation_test_detrainment_correlation_%s.png" % (figpath,vname)
plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)
