#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Integrate SSS and SST stochastic model at a single point
 - Uses output processed by get_point_data_stormtrack.py
 - Also see scrap_20230914.txt for Linux commands

Created on Thu Sep 14 15:14:12 2023

@author: gliu
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys

from tqdm import tqdm
#%% Import Custom Modules
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl

#%% User Edits

# Location
lonf           = 330
latf           = 50
locfn,loctitle = proc.make_locstring(lonf,latf)

datpath        = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/lon%s_lat%s/" % (lonf,latf)
figpath        = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20230914/"
proc.makedir(figpath)

flxs           = ["LHFLX","SHFLX","FLNS","FSNS","qnet"]
prcps          = ["PRECC","PRECL","PRECSC","PRECSL",]
varnames_ac    = ["SST","SSS"]


# Other plotting information
mons3          = proc.get_monstr(nletters=3)

#%%
def preproc_pt(ds):
    
    # Detrend (by removing ensemble mean)
    ds_dt = ds - ds.mean('ens')
    
    # Remove seasonal average
    varpt = ds_dt.values
    scycle,varpt_ds = proc.calc_clim(varpt,dim=1,returnts=True)
    varpt_dtds        = varpt_ds - scycle[:,None,:]
    return varpt_dtds

#%% Load data into dictionaries

# Get the fluxes
flxs_ds = {}
for f in range(5):
    vname = flxs[f]
    if vname == "qnet":
        ds_new = flxs_ds['FSNS'] - (flxs_ds['LHFLX'] + flxs_ds['SHFLX'] + flxs_ds['FLNS'])
        ds     = ds_new.rename(vname)
    else:
        ds = xr.open_dataset("%sCESM1_htr_%s.nc" % (datpath,vname))[vname].load() # [ens x time]
    flxs_ds[vname] = ds.copy()

#%% Get the precip fields

prcps_ds = {}
for f in range(len(prcps)):
    vname = prcps[f]
    ds = xr.open_dataset("%sCESM1_htr_%s.nc" % (datpath,vname))[vname].load() # [ens x time]
    ds = preproc_pt(ds)
    prcps_ds[vname] = ds.copy()

#%% Get the SSS and SST fields

acvars_ds = {}
for f in range(len(varnames_ac)):
    vname = varnames_ac[f]
    ds = xr.open_dataset("%sCESM1_htr_%s.nc" % (datpath,vname))[vname].load() # [ens x time]
    ds = preproc_pt(ds)
    acvars_ds[vname] = ds.copy()


#%% Visualize seasonal variability of interannual variance for the precipitation flux

mons3 = proc.get_monstr(nletters=3)

fig,axs = plt.subplots(4,1,constrained_layout=True,figsize=(6,8))

for p in range(len(prcps)):
    ax = axs[p]
    vname   = prcps[p]
    plotvar = prcps_ds[vname] # [ens x yr x mon]
    nens,nyr,nmon = plotvar.shape
    plotvar = plotvar.reshape(nens*nyr,nmon)
    for i in tqdm(range(nens*nyr)):
        ax.plot(mons3,plotvar[i,:],label="",alpha=0.15)
    mu      = np.nanmean(plotvar,(0))
    ax.plot(mons3,mu,label=vname,color="k",lw=3)
    ax.set_ylabel(vname + " (m/s)")
    ax.set_xlim([0,11])

savename = "%sCESM1_Precipitation_Flux_Anomalies.png" % figpath
plt.savefig(savename,dpi=150,bbox_inches='tight')




#%% Functions
# Note: I started writing this and gave up in the interest of time

# # Preprocess Point


    
#     # 
    
    
#     ds_dtds = proc.xrdeseason(ds_dt)
    
#     # Calculate seasonal averages

# #%% First, let's load SST and SSS

# # Get Raw SST and SSS
# sst_pt = xr.open_dataset("%sCESM1_htr_%s.nc" % (datpath,"SST")).load()
# sss_pt = xr.open_dataset("%sCESM1_htr_%s.nc" % (datpath,"SSS")).load()

# ds_raw = []
# # Deseasonalize

# ----------------------------------------------------------
#%% Retrieve the autocorrelation for SSS and SST from CESM1
# ----------------------------------------------------------
datpath_ac  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
varnames_ac = ["SST","SSS"]
ac_colors   = ["darkorange","lightseagreen"]

# Read in datasets
ds_pt = []
ac_pt = []
for v in range(2):
    ds = xr.open_dataset("%sHTR-FULL_%s_autocorrelation_thres0.nc" % (datpath_ac,varnames_ac[v]))
    
    ds  = ds.sel(lon=lonf-360,lat=latf,method='nearest').sel(thres="ALL").load()# [ens lag mon]
    ds_pt.append(ds)
    ac_pt.append(ds[varnames_ac[v]].values) # [variable][ens x lag x month]

#%% AC Plot

kmonth   = 1
lags     =  np.arange(0,37,1)
xtk2     = lags[::3]

tickfreq = 2
title    = None
usegrid  =True
yticks   = np.arange(-.2,1.2,0.2)

sameplot = True # True to plot SSS and SST on separate plots

if sameplot:
    fig,ax   = plt.subplots(1,1,figsize=(6,4.5),constrained_layout=True)
else:
    fig,axs   = plt.subplots(2,1,figsize=(8,8),constrained_layout=True)
for v in range(2):
    
    if not sameplot:
        ax       = axs[v]
        title    = "%s Autocorrelation @ %s, CESM1" % (varnames_ac[v],loctitle,)
    else:
        title    = "Autocorrelation @ %s, CESM1" % (loctitle,)
    
    if not sameplot or v == 0:
        ax,ax2   = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title,usegrid=usegrid,
                                 tickfreq=tickfreq)
        
    ax.set_yticks(yticks,minor=True)
    ax.set_ylim([yticks[0],yticks[-1]])
    
    for e in range(42):
        plot_ac = ac_pt[v][e,:,kmonth]
        ax.plot(lags,plot_ac,alpha=0.1,color=ac_colors[v],lw=2,label="",zorder=3)
    ax.plot(lags,np.nanmean( ac_pt[v][:,:,kmonth],0),color=ac_colors[v],lw=2,label=varnames_ac[v] +" (ens. mean)" )
    
    if sameplot:
        ax.legend()
    
#%% Set up the stochastic model, by getting needed parameters

# Set some constants ----------------------------------------------------------
dt         = 3600*24*30 # Timestep [s]
cp         = 3850       # 
rho        = 1026       # Density [kg/m3]
B          = 0.2        # Bowen Ratio, from Frankignoul et al 1998
L          = 2.5e6      # Specific Heat of Evaporation [J/kg], from SSS model document

# Get MLD seasonal cycle and convert to meters --------------------------------
mld       = xr.open_dataset("%sCESM1_htr_%s.nc" % (datpath,"HMXL")).HMXL.load().values # [ens x time]
hpt       = proc.calc_clim(mld,1)/100
h_emean   = hpt.mean(0)
h_smean   = hpt.mean() # ensemble and seasonal mean, 72.65499 [m]

# Compute Entrainment/Detrainment times ---------------------------------------
kprev_ens    = np.array([scm.find_kprev(hpt[e,:])[0] for e in range(40)])
kprev_mean,_ = scm.find_kprev(hpt.mean(0)) 

# Get mean salinity -----------------------------------------------------------
sss_pt    = xr.open_dataset("%sCESM1_htr_%s.nc" % (datpath,"SSS")).SSS.load().values # [ens x time]
sbar_mon      = proc.calc_clim(sss_pt,dim=1)
sbar_ens  = np.nanmean(sss_pt,1) # Mean for each ensemble 
sbar      = np.nanmean(sbar_ens) # Overall mean SSS. This was 36.5 [psu] in Frankignoul et al 1998

# Retrieve damping values -----------------------------------------------------
dampings_pt = []
for f in range(5):
    vname      = flxs[f] + "_damping"
    hff        = xr.open_dataset("%sCESM1_htr_%s.nc" % (datpath,vname))[vname].load().values # [ens x mon x lag]
    dampings_pt.append(hff)

# Net Heat Flux Damping
lbd_qnet       = dampings_pt[-1].mean(0)[...,0] # Taken ensemble mean and first lag
lbd_qnet_smean = lbd_qnet.mean()          # Take seasonal mean 19.9 [W/m2/degC]

# Latent Heat Flux Damping
lbd_lhf       = dampings_pt[0].mean(0)[...,0] # Taken ensemble mean and first lag
lbd_lhf_smean = lbd_lhf.mean()          # Take seasonal mean 14.1249 [W/m2/degC]

# Sensible Heat Flux Damping
lbd_shf       = dampings_pt[1].mean(0)[...,0] # Taken ensemble mean and first lag
lbd_shf_smean = lbd_shf.mean()                # Take seasonal mean 4.475 [W/m2/degC]

## Get seasonal values of F' ---------------------------------------------------


#%% First test: the SST-Evaporation on SSS feedback.
"""

Is (lambda_e * T') == (lambda_a * S')?

"""

# Make a white noise timeseries
nyrs   = 10000
eta    = np.random.normal(0,1,12*nyrs)

# Set some universal variables

h = h_emean #h_smean * np.ones(12)

#%% Do Unit Conversions (for temperature) ---------------------------------------
#h      = h_smean*np.ones(12)
hff    = lbd_qnet_smean*np.ones(12)
lbd_a  = hff / (rho*cp*h) * dt
alpha  = 1*np.ones(12)
F      = np.tile(alpha,nyrs) * eta

# First integrate temperature -------------------------------------------------
T      = scm.integrate_entrain(h[None,None,:],
                               kprev_mean[None,None,:],
                               lbd_a[None,None,:],
                               F[None,None,:],
                               T0=np.zeros((1,1)),
                               multFAC=True,
                               debug=False,
                               Td0=False)



#%% Now try the version lambda_a * S' -----------------------------------------
# Note that this is essentially the same as above, but with weaker damping
# The forcing term should theoretically be reesimated but I didnt...
#h      = h_smean*np.ones(12)
hff    = lbd_lhf_smean*np.ones(12)
lbd_a  = hff / (rho*cp*h) * dt
alpha  = 1*np.ones(12)
F      = np.tile(alpha,nyrs) * eta
# Integrate for S1 (Salinity Method 1)
S1      = scm.integrate_entrain(h[None,None,:],
                               kprev_mean[None,None,:],
                               lbd_a[None,None,:],
                               F[None,None,:],
                               T0=np.zeros((1,1)),
                               multFAC=True,
                               debug=False,
                               Td0=False)

#%% Now do the version where we use lbd_e*T' ----------------------------------
hff    = lbd_lhf_smean*np.ones(12)
lbd_a  = hff / (rho*cp*h) * dt
lbd_e  = (sbar*cp*lbd_a)/ (L*(1+B))
add_F  = np.tile(lbd_e,nyrs) * T[0,0,:] *-1# lbd_e * T'
lbd_a  = np.array([0,])

# Integrate for S2 (Salinity Method 2)
S2      = scm.integrate_entrain(h[None,None,:],
                               kprev_mean[None,None,:],
                               lbd_a[None,None,:],
                               F[None,None,:],
                               T0=np.zeros((1,1)),
                               multFAC=True,
                               debug=False,
                               Td0=False,
                               add_F=add_F[None,None,:])

#%% Calculate the autocorrelation

output_ts = [T,S1,S2]
output_ts = [ts.squeeze() for ts in output_ts]


acs_out,cfs_out = scm.calc_autocorr(output_ts,lags,kmonth,calc_conf=True,)
#config = {'lbd'}

#%% Make a plot

sm_acs_name  = ["T (SM)","S($\lambda_{LHF} S'$)","S ($\lambda_e T'$)"]
sm_acs_color = ["indianred","indigo","violet"]


kmonth   = 1
lags     =  np.arange(0,37,1)
xtk2     = lags[::3]

tickfreq = 2
title    = None
usegrid  =True
yticks   = np.arange(-.2,1.2,0.2)

sameplot = True # True to plot SSS and SST on separate plots

if sameplot:
    fig,ax   = plt.subplots(1,1,figsize=(6,4.5),constrained_layout=True)
else:
    fig,axs   = plt.subplots(2,1,figsize=(8,8),constrained_layout=True)
for v in range(2):
    
    if not sameplot:
        ax       = axs[v]
        title    = "%s Autocorrelation @ %s, CESM1" % (varnames_ac[v],loctitle,)
    else:
        title    = "Autocorrelation @ %s, CESM1" % (loctitle,)
    
    if not sameplot or v == 0:
        ax,ax2   = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title,usegrid=usegrid,
                                 tickfreq=tickfreq)
        
    ax.set_yticks(yticks,minor=True)
    ax.set_ylim([yticks[0],yticks[-1]])
    
    for e in range(42):
        plot_ac = ac_pt[v][e,:,kmonth]
        ax.plot(lags,plot_ac,alpha=0.1,color=ac_colors[v],lw=2,label="",zorder=3)
    ax.plot(lags,np.nanmean( ac_pt[v][:,:,kmonth],0),color=ac_colors[v],lw=2,label=varnames_ac[v] +" (ens. mean)" )
    
    # if sameplot:
    #     ax.legend()

for s in range(3):
    ax.plot(lags,acs_out[s],label=sm_acs_name[s],ls='dashed',c=sm_acs_color[s],lw=2)
ax.legend()

#%% Visualize seasonality in S-bar

fig,ax = plt.subplots(1,1,figsize=(6,4))
for e in range(42):
    ax.plot(mons3,sbar_mon[e,:],color="gray",alpha=0.2)
ax.plot(mons3,sbar_mon.mean(0),color="k",alpha=1)
ax.set_xlabel("Month")
ax.set_ylabel("Mean SSS (psu)")
ax.set_xlim([0,11])
savename = "%sCESM1_SSS_bar.png" % figpath
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Compute the precipitation flux term

#sbar_mon # [ens mon z_t]

prcp_total = np.zeros((nens,86,nmon))
for p in prcps:
    prcp_total += prcps_ds[p]
    
# Compute the forcing term
p_forcing = (sbar_mon.squeeze()[:40,None,:] * prcp_total[:40,:,:])/hpt[:40,None,:] # [ens x yr x mon]







