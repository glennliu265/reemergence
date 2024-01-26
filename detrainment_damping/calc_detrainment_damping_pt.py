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
tsanom = tsmonyr - scycle[:, None, :, :]

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

# Functions
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
    
    # Compute ACF
    acfs_mon = calc_acf_ens(tsens,lags) # [mon x lag x depth]
    
    # Compute Expfit
    tau_est,acf_est = fit_exp_ens(acfs_mon,lagmax) # [mon x depth], [mon x lags x depth]
    
    # Compute Detrainment Damping
    kprev,_ = scm.find_kprev(hclim_ens)
    lbd_d   = scm.calc_tau_detrain(hclim_ens,kprev,z,tau_est,debug=False)
    
    # Save Output
    lbd_d_all[e,:]       = lbd_d.copy()
    tau_est_all[e,:,:]   = tau_est.copy()
    acf_est_all[e,:,:,:] = acf_est.copy()
    acf_mon_all[e,:,:,:] = acfs_mon.copy()
    
    # End Ens Loop


#%% Save output

#%%

# --------------------------------------------------------------------
#%% 3. Compute Autocorrelation at each level
# --------------------------------------------------------------------




#%% Compute detrainment damping for each month

# Plot Settings
debug   = False
lm      = 2
lcolors = ["violet","orange","blue"]
taudts = []

# Compute Detrainment Damping
for lm in range(3):
    taudt = scm.calc_tau_detrain(hclim.mean(1),kprev,z,tau_est[lm,...],debug=debug)
    taudts.append(taudt)

#% Visualize Detrainment -----------------------------------------------------
kprevround = [int(np.round(k)) for k in kprev]
idhmax     = np.argmin(np.abs(z-hmax))
tau_hmax   = np.array([tau_est[lm,kprevround[im],idhmax] for im in range(12)])#tau_est[lm,:,idhmax]


tau_hmax[kprev==0.] = 0

fig,ax =viz.init_monplot(1,1)
for lm in range(3):
    ax.plot(mons3,np.abs(taudts[lm]),label=r"Monthy $\lambda^d$ (%i-lag fit)" % (lagmaxes[lm]),
            marker="o",c=lcolors[lm])
ax.plot(mons3,np.abs(tau_hmax),label="Max Depth (h=%.2f)"% (z[idhmax]),marker="s",color='limegreen')
#ax.plot(mons3,tau_detrain_nonfunction,label="Monthly Sd, In Script",ls='dotted') # Confirmed that function was ok!
ax.legend()
ax.set_title("Estimated $\lambda^d$ (Ens %02i, Exp fit) @ %s" % (e+1,loctitle))
ax.set_ylabel("e-folding timescale ($month^{-1}$)")
savename = "%slbdtau_estimate_monthlycomparison_%s_lagmax%02i_ens%02i.png" % (figpath,locfn,lagmaxes[lm],e+1)
plt.savefig(savename,dpi=150,bbox_inches='tight')

