#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Empirically estimate damping of detrained anomalies (Td', Sd')
at a single point in CESM1-LENs Historical.

Works with output from 
- repair_file_SALT_CESM1.py
- get-pt-data-stormtrack (legacy?)

To Do:
- Rework to be compatible with extract_file_loop

Copied from Td Sd decay vertical on 2024.01.25
Created on Thu Jan 25 23:08:42 2024

"""

from amv import proc, viz
import amv.loaders as dl
import scm
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
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20240126/"
proc.makedir(figpath)
outpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/%s/" % locfn

# Other toggles
debug = True # True to make debugging plots

# Plotting Stuff
mons3 = proc.get_monstr(nletters=3)

# --------------------------------------------------------------------
#%% 1. Load necessary files (see repair_file_SALT_CESM1.py)
# --------------------------------------------------------------------

# Load SALT ----------------------------------------------------------------
# Paths and Names
ncsalt  = outpath + "CESM1_htr_SALT_repaired.nc"
ds_salt = xr.open_dataset(ncsalt)

# Load
z       = ds_salt.z_t.values  # /100 NOTE cm --> meter conversion done in repair code
times   = ds_salt.time.values
salt    = ds_salt.SALT.values  # [Ens x Time x Depth ]
nens, ntime, nz = salt.shape

# Get strings for time
timesstr = ["%04i-%02i" % (t.year, t.month) for t in times]

# Get Ensemble Numbers
ens     = np.arange(nens)+1

# Load HBLT ----------------------------------------------------------------
# Paths and Names
mldpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/CESM_proc/"
mldnc   = "HBLT_FULL_HTR_lon-80to0_lat0to65_DTFalse.nc"

# Load and select point
dsh         = xr.open_dataset(mldpath+mldnc)
hbltpt      = dsh.sel(lon=lonf-360, lat=latf,
                 method='nearest').load()  # [Ens x time x z_t]

# Compute Mean Climatology [ens x mon]
hclim       = hbltpt.groupby('time.month').mean('time').squeeze().HBLT.values/100  # Ens x month, convert cm --> m

# Compute Detrainment month
kprev, _    = scm.find_kprev(hclim.mean(1)) # Detrainment Months
hmax        = hclim.mean(1).max() # Maximum MLD of seasonal cycle

# --------------------------------------------------------------------
#%% 2. Preprocessing (Deseason and Detrend)
# --------------------------------------------------------------------
# Note, there should be no NaN values, accomplished through the "repair" script.

# 2A. Compute the seasonal cycle and monthly anomaly
# Get Seasonal Cycle
scycle, tsmonyr = proc.calc_clim(salt, 1, returnts=True)  # [ens x yr x mon x z]

# Compute monthly anomaly
tsanom        = tsmonyr - scycle[:, None, :, :]

# 2B. Remove the ensemble average

tsanom_ensavg = np.nanmean(tsanom, 0)

tsanom_dt     = tsanom - tsanom_ensavg[None, ...]

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


#%% Perform the Loop

# Functions ---
def calc_acf_ens(tsens,lags):
    # tsens is the anomalized values [yr x mon x z]
    acfs_mon = []
    for im in range(12):
        basemonth   = im+1
        varin       = tsanom_dt[e, ...].transpose(1, 0, 2)  # Month x Year x Npts
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

# ---------------------

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
    acfs_mon = calc_acf_ens(tsens,lags) # [mon x lag x depth]
    
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


#%% Save output

savename = "%sSd_damping_CESM1_HTR_FULL_%s_HBLT_%ilagfig_lags%02i.npz" % (outpath,locfn,lagmax,lags[-1])
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

# <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#%% Check Output 
# <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>

# Load npz output
savename = "%sSd_damping_CESM1_HTR_FULL_%s_HBLT_%ilagfig_lags%02i.npz" % (outpath,locfn,lagmax,lags[-1])
ld       = np.load(savename,allow_pickle=True)

# Load some variables
lbd_d   = ld['lbd_d']
tau_est = ld['tau_est']
hclim   = ld['hblt']


# Make Figure
fig,axs =viz.init_monplot(2,1,figsize=(8,6))

ax = axs[0]
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

ax = axs[1]
for e in range(nens):
    plotvar = hclim[:,e]
    if e == 0:
        lab="Indv. Member"
    else:
        lab=""
    ax.plot(mons3,plotvar,label=lab,color="gray",alpha=0.25)
ax.plot(mons3,hclim.mean(1),label="Ens. Avg.",c="k")
ax.set_ylabel("MLD (meters)")

savename = "%sSd_damping_AllEns_%s_%ilagfig_lags%02i.png" % (figpath,locfn,lagmax,lags[-1])
plt.savefig(savename,dpi=150,bbox_inches='tight')
