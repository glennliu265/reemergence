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

# --------------------------------------------------------------------
#%% 1. Load necessary files (see repair_file_SALT_CESM1.py)
# --------------------------------------------------------------------

# Load SALT ----------------------------------------------------------------
# Paths and Names
vname = "SALT"
if vname == "TEMP":
    ncname = "CESM1_htr_TEMP_repaired.nc"
else:
    ncname  = "CESM1_htr_SALT_repaired.nc"
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
    

# --------------------------------------------------------------------
#%% 2. Preprocessing (Deseason and Detrend)
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

# -------------------------------------------------
#%% Method (1) Compute depth-dependent detrainment
# -------------------------------------------------

# Functions ---
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

#%% Save output (depth dependent entrainment)


if vname == "TEMP":
    outvar = "T"
elif vname == "SALT":
    outvar = "S"

savename = "%s%sd_damping_CESM1_HTR_FULL_%s_HBLT_%ilagfig_lags%02i.npz" % (outpath,outvar,locfn,lagmax,lags[-1])
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
savename = "%s%sd_damping_CESM1_HTR_FULL_%s_HBLT_%ilagfig_lags%02i.npz" % (outpath,outvar,locfn,lagmax,lags[-1])
ld       = np.load(savename,allow_pickle=True)

# Load some variables
lbd_d   = ld['lbd_d']
tau_est = ld['tau_est']
hclim   = ld['hblt']

# Make Figure
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
    #ax.plot(np.arange(1,13,1),plotvar,label=lab,color="gray",alpha=0.25,zorder=-1)
    ax.plot(np.arange(0,12,1),plotvar,label=lab,color="gray",alpha=0.25,zorder=-1)
    
hmu     = hclim.mean(1)
# kprev,_ = scm.find_kprev(hmu,)
# ax      = viz.viz_kprev(hclim.mean(1),kprev,)

ax.set_xlim([0,11])
ax.plot(mons3,hclim.mean(1),label="Ens. Avg.",c="k")



ax.set_ylabel("MLD (meters)")

savename = "%sSd_damping_AllEns_%s_%ilagfig_lags%02i.png" % (figpath,locfn,lagmax,lags[-1])
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Plot example detrainment damping steps


# First, plot timeseries at a given timepoint
e = 0

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

#%% Plot the Timeseries
im        = 10
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

ax.vlines(np.arange(dtid,len(tsens[:,:,iz].flatten()),12),ymin=-1,ymax=1,linewidths=1,zorder=-1)

#%% Plot the ACF and Fit

iz = 0

for iz in range(len(z)):
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
    plt.savefig(savename,dpi=150,bbox_inches='tight',transparent=True)

#%%



#%% Debugging/Troubleshooting... Let's manually do this and check our work


# First Load a file computed with the other script

path2 = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/ocn_var_3d/"
nc2   = "CESM1_HTR_FULL_lbd_d_params_TEMP_detrendensmean_lagmax3_ens01_regridNN.nc"
ds2   = xr.open_dataset(path2+nc2)

acfpt_script = ds2.acf_mon.sel(lon=lonf-360,lat=latf,method='nearest').load()


# with the timeseries above

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

#%%


plotvar = tsens[:,:,iz].flatten()



savename = "%sLbdd_Demo_TEMP_Timeseries_z%03i.png" % (figpath,z[iz])



# -------------------------------------------------
#%% Method (2/3) Surface or Depth Dependent Corr (Detrain, Entrain)
# -------------------------------------------------


#% Recompute detrainment damping based on the month of detrainment
ts_surface   = tsanom[:,:,:,0] # ens x yr x mon 
nyr          = tsanom.shape[1]
surface      = False # Set to True for Method (2), False for Method (3)

expf3        = lambda t,b: np.exp(b*t)         # No c and A
corr_byens   = np.zeros((nens,12))
tau_byens    = np.zeros((nens,12))
acffit_byens = np.zeros((nens,12,nlags)) 

# Plotting Stuff
xtks = np.arange(0,37,1)
#lagxall = np.zeros((2,nens,12))


for im in tqdm(range(12)):
    
    # Get Index of the detraining month
    detrain_mon = kprev[im]
    dtid        = int(np.floor(detrain_mon) - 1) # Overestimate to compensate for MLD variability
    
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
        entrain_mon = im+12
    
    x    = [detrain_mon,entrain_mon]
    xlag = [0,entrain_mon-detrain_mon]
    
    # Just apply shift here
    if surface: # Take the surface values
        
        detrain_anom = ts_surface[:,:(nyr-shift),dtid] # {Ens x Year}
        entrain_anom = ts_surface[:,shift:,im] # {Ens x Year}
    
    else: # Take from Entraining Depth
        hdetrain = hclim[dtid,e]
    
    corr_allens = []
    for e in range(nens):
        
        if surface:
            x1 = ts_surface[e,:(nyr-shift),dtid] # Detrain Anoms {Ens x Year}
            x2 = ts_surface[e,shift:,im]         # Entrain Anoms {Ens x Year}
        else:        
        
        corr_ens         = np.corrcoef(x1,x2)[0,1]
        corr_byens[e,im] = corr_ens
        
        y                    = [1,corr_ens]
        
        fitout               = proc.expfit(np.array(y),np.array(xlag),1)
        
        tau_byens[e,im]      = fitout['tau_inv']
        acffit_byens[e,im,:] = expf3(lags,fitout['tau_inv'])
        
        if debug:

            use_script = False # Set to True to use ACF computed from other scriptd
            # Get ACFs from above and below the level
            hdetrain = hclim[dtid,e]
            hid_0    = np.argmin(np.abs(z - hdetrain))
            if use_script:
                acf0     = acfpt_script.isel(mon=im,z_t=iz)#acf_mon_all[e,dtid,:,hid_0]
                acf1     = acfpt_script.isel(mon=im,z_t=iz+1)#acf_mon_all[e,dtid,:,hid_0+1]
                acf_surf = acfpt_script.isel(mon=im,z_t=0)
            else:
                acf0     = acf_mon_all[e,dtid,:,hid_0]
                acf1     = acf_mon_all[e,dtid,:,hid_0+1]
                acf_surf = acf_mon_all[e,im,:,0]
            
            
            
            fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(8,4))
            title = "%s ACF Fit (Entrain Mon: %02i, Detrain Mon: %02i, Ens %02i)" % (vname,im+1,detrain_mon,e+1)
            ax,ax2  = viz.init_acplot(dtid,xtks,lags,title=title,ax=ax)
            ax.plot(lags,expf3(lags,fitout['tau_inv']),label=r"Exp Fit $\tau$=%.2f Months" % (1/np.abs(fitout['tau_inv'])))
            #ax.plot(lags,acf_surf,label="ACF (lag0=%02i, z=%.2f)" % (im+1,z[0]),color="orange")
            
            
            #ax.plot(lags,acf0,label="ACF (lag0=%02i, z=%.2f)" % (dtid+1,z[hid_0]),color="darkblue")
            #ax.plot(lags,acf1,label="ACF (lag0=%02i, z=%.2f)" % (dtid+1,z[hid_0+1]),color="magenta")
            
            ax.scatter(xlag,y,ls="None",marker="d",c='k',label="Detrain Corr: %.2f" % (corr_ens))
            ax.legend()
            ax.set_xticks(xtks)
            ax.set_xlim([0,24])
            ax2.set_xlim([0,24])
            
            savename = "%sSurface_Entrain_Estimates/EntrainMon%02i_Ens%02i" % (figpath,im+1,e+1)
            plt.savefig(savename,dpi=150,bbox_inches='tight')
        #     plt.close()
#%% Save tau estimates for use in another script...

# Do some final preprocessing

# Save Surface Estimates
tau_byens   = np.abs(tau_byens)
coords      = dict(ens=np.arange(1,43,1),mon=np.arange(1,13,1))
da_tau2     = xr.DataArray(tau_byens,coords=coords,dims=coords,name='lbd_d')
savenametau = "%sLbdd_estimate_surface_%s.nc" % (outpath,vname)
da_tau2.to_netcdf(savenametau)

# Save Deep Estimates
tau_3d      = np.abs(lbd_d_all)
da_tau3d    = xr.DataArray(tau_3d,coords=coords,dims=coords,name='lbd_d')
savenametau = "%sLbdd_estimate_deep_%s.nc" % (outpath,vname)
da_tau3d.to_netcdf(savenametau)

#%% Plot Surface vs. 3D Detrainment Estimates (Scatter, Ens Mean, and 1 Stdev)
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

#%%



#%% Plot Surface vs. 3D Detrainment Estimates (Scatter, Ens Mean, and 1 Stdev)
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

#%% Plot the correlations

def monstacker(scycle):
    return np.hstack([scycle,scycle[:1]])
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
    
    
    