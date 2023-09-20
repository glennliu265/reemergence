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

# 
Fori  = np.array([53.36403275, 50.47200521, 43.19549306, 32.95324516, 26.30336189,
           22.53761546, 22.93124771, 26.54155223, 32.79647001, 39.71981049,
           45.65141678, 50.43875758])
Ftest = np.ones(12)*53.36403275

# Other plotting information
mons3          = proc.get_monstr(nletters=3)

#%% Preprocessing Function

def preproc_pt(ds):
    
    # Detrend (by removing ensemble mean)
    ds_dt = ds - ds.mean('ens')
    
    # Remove seasonal average
    varpt             = ds_dt.values
    scycle,varpt_ds = proc.calc_clim(varpt,dim=1,returnts=True)
    varpt_dtds        = varpt_ds - scycle[:,None,:]
    return varpt_dtds

def init_annplot(figsize=(6,4),ax=None,figflag=False):
    if ax is None:
        figflag=True
        fig,ax = plt.subplots(1,1,figsize=figsize,constrained_layout=True)
    ax.set_xlim([0,11])
    ax.set_xticks(np.arange(0,12))
    ax.set_xticklabels(mons3)
    ax = viz.add_ticks(ax)
    if figflag:
        return fig,ax
    else:
        return fig

#%% Get old data information (from synth_stocmod)
Fpt = np.array([55.278503, 53.68089 , 42.456623, 33.448967, 22.954145, 22.506973,
       22.151728, 24.135042, 33.337887, 40.91648 , 44.905064, 51.132706])
mldpt = np.array([127.89892564, 143.37650773, 113.72955319,  61.17662532,
        29.92363254,  22.41539899,  23.2999003 ,  27.15075052,
        36.19567419,  50.82827159,  71.20451299,  97.11546462])

damppt = np.array([18.79741573, 12.31693983,  9.71672964,  8.03764248,  7.99291682,
        6.53819919,  6.33767891,  8.54040241, 13.54183531, 19.00482941,
       22.59606743, 22.18767834])
kprevpt = np.array([2.52206314, 2.        , 0.        , 0.        , 0.        ,
       0.        , 5.88219582, 5.36931217, 4.79931389, 4.33111561,
       3.80918499, 3.31614011])


cesmauto = np.array([1.        , 0.87030124, 0.70035406, 0.56363062, 0.39430458,
       0.31124585, 0.29375811, 0.27848886, 0.31501385, 0.38280692,
       0.42196468, 0.46813908, 0.4698927 , 0.40145464, 0.32138829,
       0.27078197, 0.19005614, 0.15290666, 0.16341374, 0.15956728,
       0.17240383, 0.1911096 , 0.2213633 , 0.25900566, 0.25901782,
       0.24166312, 0.17903541, 0.14626024, 0.09599903, 0.06320866,
       0.06668568, 0.10627938, 0.10912452, 0.10387345, 0.08380565,
       0.09628464, 0.09792761])

oldintegration = np.array([1.        , 0.9076469 , 0.76167929, 0.59542029, 0.42233226,
       0.32332272, 0.27163913, 0.27748115, 0.31971224, 0.3857661 ,
       0.46209719, 0.53502147, 0.54044438, 0.48706086, 0.39984809,
       0.31327671, 0.21721238, 0.17423834, 0.14649725, 0.14413544,
       0.16182631, 0.18972564, 0.23529796, 0.27853343, 0.27649187,
       0.25082504, 0.21038523, 0.16064941, 0.1202002 , 0.10343898,
       0.09064742, 0.08737261, 0.08828676, 0.103625  , 0.12602822,
       0.14773722, 0.1504384 ])


# ><><><><><><><><><><><><><><><><><><><><>< ><><><><><><><><><><><><><><><><><><><><><
#%% Parameter Loading Section
# ><><><><><><><><><><><><><><><><><><><><>< ><><><><><><><><><><><><><><><><><><><><><
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
ptot_full = [] 
ptot      = []
for f in range(len(prcps)):
    vname = prcps[f]
    ds = xr.open_dataset("%sCESM1_htr_%s.nc" % (datpath,vname))[vname].load() # [ens x time]
    
    # Get the full field
    ptot_full.append(ds.values)
    
    # Compute anomalies sand save
    ds = preproc_pt(ds)
    prcps_ds[vname] = ds.copy()
    ptot.append(ds.copy())

ptot = np.array(ptot)
ptot = ptot.sum(0) # [Ens x Time X Mon]

#%% Get the SSS and SST fields
acvars_ds = {}
acvars_np = []
for f in range(len(varnames_ac)):
    vname = varnames_ac[f]
    ds = xr.open_dataset("%sCESM1_htr_%s.nc" % (datpath,vname))[vname].load() # [ens x time]
    ds = preproc_pt(ds)
    acvars_ds[vname] = ds.copy() # [ens x year x mon]
    acvars_np.append(ds)


#%% Check how the stdev(SST,SSS) each season looks like (across ensemble member)
fig,axs = plt.subplots(2,1,constrained_layout=True)
for v in range(2):
    ax      = axs[v]
    vname   = varnames_ac[v]
    for e in range(42):
        plotvar = acvars_ds[vname][e,:,:].std(0) # 
        ax.plot(mons3,plotvar,alpha=0.2,label="")
    
    plotvar = acvars_ds[vname][:,:,:].std(1).mean(0) # 
    ax.plot(mons3,plotvar,alpha=1,label="Ens. Mean",color="k",lw=1)
    ax.set_ylabel("%s stdev" % (vname))

plt.suptitle("SST and SSS stdev (year)")
savename = "%sCESM1_SST_SSS_stdev_Anomalies.png" % figpath
plt.savefig(savename,dpi=150,bbox_inches='tight')

# <o><o><o><o><o><o><o><o><o><o><o><o><o><o>
#%% Do same for annual values of SST
# <o><o><o><o><o><o><o><o><o><o><o><o><o><o>
fig,axs = plt.subplots(2,1,constrained_layout=True,figsize=(6,4))
for p in range(2):
    ax = axs[p]
    vname   = varnames_ac[p]
    plotvar = acvars_ds[vname].squeeze() # [ens x yr x mon]
    nens,nyr,nmon = plotvar.shape
    plotvar = plotvar.reshape(nens*nyr,nmon)
    for i in tqdm(range(nens*nyr)):
        ax.plot(mons3,plotvar[i,:],label="",alpha=0.15)
    mu      = np.nanmean(plotvar,(0))
    ax.plot(mons3,mu,label=vname,color="k",lw=3)
    ax.set_ylabel(vname)
    ax.set_xlim([0,11])
plt.suptitle("SST and SSS values for each year and ensemble")
savename = "%sCESM1_SST_SSS_Anomalies.png" % figpath
plt.savefig(savename,dpi=150,bbox_inches='tight')
# <o><o><o><o><o><o><o><o><o><o><o><o><o><o>
#%% Plot seasonal cycle of Interann Precip Variability
# <o><o><o><o><o><o><o><o><o><o><o><o><o><o>
#% Visualize seasonal variability of interannual variance for the precipitation flux
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

# <o><o><o><o><o><o><o><o><o><o><o><o><o><o>
#%% Plot seasonal cycle of Precip
# <o><o><o><o><o><o><o><o><o><o><o><o><o><o>
# Quickly check the seasonality to make sure that the sign is correct

ptot_full   = np.array(ptot_full) # [4 x 42 x 1032]
#ptot_full = ptot_full.sum(0) # [Ens x Time X Mon]
ptot_scycle = proc.calc_clim(ptot_full,2)

fig,axs     = plt.subplots(4,1,constrained_layout=True,figsize=(6,8))
for p in range(len(prcps)):
    
    ax      = axs[p]
    vname   = prcps[p]
    for e in tqdm(range(nens)):
        ax.plot(mons3,ptot_scycle[p,e,:],label="",alpha=0.15)
    
    mu      = np.nanmean(ptot_scycle[p,:,:],(0))
    ax.plot(mons3,mu,label=vname,color="k",lw=1.5)
    ax.set_ylabel(vname + " (m/s)")
    ax.set_xlim([0,11])
    
#ax.plot(mons3,ptot_scycle[])
savename = "%sCESM1_Precipitation_Flux_Scycle.png" % figpath
plt.savefig(savename,dpi=150,bbox_inches='tight')


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
    
    

    # Old Stuff
    if v == 0:
        ax.plot(lags,cesmauto,color="k",label='CESM1 PiControl')
        ax.plot(lags,oldintegration,color='cornflowerblue',label="Old Stochastic Model")
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

#%%

## Estimate Precipitation forcing
ptot_mon   = np.nanstd(ptot,(0,1))
ptot_force = (ptot_mon * sbar_mon.mean(0).squeeze()) * dt / h_emean

fig,ax      = plt.subplots(1,1)

ax.plot(mons3,ptot_force,)
ax.set_ylabel("Precip. Forcing (psu/mon)")
ax.set_title(r"Precipitation Forcing ($\frac{ \overline{S} P'}{\rho h}$) @ %s" % (loctitle))
ax.set_xlim([0,11])
savename = "%sCESM1_Precipitation_Flux_Scycle.png" % figpath
plt.savefig(savename,dpi=150,bbox_inches='tight')

# ><><><><><><><><><><><><><><><><><><><><>< ><><><><><><><><><><><><><><><><><><><><><
#%% Model Section
# ><><><><><><><><><><><><><><><><><><><><>< ><><><><><><><><><><><><><><><><><><><><><

#%% First test: the SST-Evaporation on SSS feedback.
"""

Is (lambda_e * T') == (lambda_a * S')?

"""

# Make a white noise timeseries
nyrs   = 10000
eta    = np.random.normal(0,1,12*nyrs)

# Indicate if you have seasonal variation
svary  = True

# Set some universal variables
if svary:
    h      = h_emean#mldpt#h_emean # [mon]
else:
    h      = h_emean #mldpt * np.ones(12)#h_emean.mean() * np.ones(12)

#%% Do Unit Conversions (for temperature) ---------------------------------------
#h      = h_smean*np.ones(12)
if svary:
    hff    = lbd_lhf# lbd_qnet
    Fmag   = Fori     # Forcing magnitude
else:
    hff    = lbd_qnet.mean()*np.ones(12)
    Fmag   = Fori.mean()*np.ones(12)
lbd_a  = hff / (rho*cp*h) * dt
alpha  = Fmag / (rho*cp*h) * dt
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

# #%% Peak at the params

# fig,ax = init_annplot()
# ax.plot(mons3,lbd_a,label="Damping",color="orange")
# ax.plot(mons3,alpha,label="Forcing",color="limegreen")
# ax.set_ylabel("deg C / mon")
# ax.set_title("Stochastic Model Inputs (T')")
# ax.legend()
#%% Now try the version lambda_a * S' -----------------------------------------
# Note that this is essentially the same as above, but with weaker damping
# The forcing term should theoretically be reesimated but I didnt...


# lbd_lhf_cust = np.array([16.96781049, 13.30814792, 5, 5, 11.5776733 ,
#        10.75794417, 11.08759759, 12.94189574, 14.99567006, 17.29924383,
#        18.58622274, 18.9057527 ])

h = np.array([137.26482 , 154.96883 , 150, 150,  100,
        25,  23, 26 ,  30,  51.93153 ,
        73.71345 , 102.56731 ])


lbd_lhf_cust = np.array([16.96781049, 13.30814792, 11, 11, 0,
        0, 0, 0, 0, 0,
        18.58622274, 18.9057527 ])

ptot_force_cust = np.array([0.0098616 , 0.00881426, 0.00973901, 0.01416393, 0.0270726 ,
       0.03760537, 0.03685692, 0.03333382, 0.03046539, 0.02448979,
       0.01853393, 0.01280091])


if svary:
    hff    = lbd_lhf#_cust
    alpha  = ptot_force.mean() * np.ones(12)#_cust
else:
    hff    = lbd_lhf.mean() * np.ones(12)
    alpha  = ptot_force.mean() * np.ones(12)

lbd_a  = hff / (rho*cp*h) * dt
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




if svary:
    hff    = lbd_lhf_cust
else:
    hff    = lbd_lhf.mean() * np.ones(12)
    
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

# ><><><><><><><><><><><><><><><><><><><><>< ><><><><><><><><><><><><><><><><><><><><><
#%% Analysis Section
# ><><><><><><><><><><><><><><><><><><><><>< ><><><><><><><><><><><><><><><><><><><><><
#% Calculate, plot and compare the ACF Calculate the autocorrelation --------

kmonth      = 1
lags        =  np.arange(0,37,1)
recalc_cesm = True

# Plotting Options
xtk2     = lags[::3]
tickfreq = 2
title    = None
usegrid  =True
yticks   = np.arange(-.2,1.2,0.2)
sameplot = True # True to plot SSS and SST on separate plots

# Calculate ACF and make labels
output_ts       = [T,S1,S2]
output_ts       = [ts.squeeze() for ts in output_ts]
acs_out,cfs_out = scm.calc_autocorr(output_ts,lags,kmonth,calc_conf=True,)
sm_acs_name     = ["T (SM)","S($\lambda_{LHF} S'$)","S ($\lambda_e T'$)"]
sm_acs_color    = ["indianred","indigo","violet"]

# (Re) calculate mean ACF
if recalc_cesm:
    nlags    =  len(lags)
    acf_cesm = []
    for v in range(2):
        invar = acvars_np[v].squeeze()
        invar[np.isnan(invar)] = 0
        invars_list = [invar[e,...].flatten() for e in range(42)]
        acs_cesm,cfs_cesm = scm.calc_autocorr(invars_list,lags,kmonth,calc_conf=True)
        acf_cesm.append(acs_cesm)
        #acs_cesm = np.array([print(ac.shape) for ac in acs_cesm])
        
        
# Make the plot
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
    
    if recalc_cesm:
        ncount = 0
        meanacf = np.zeros((nlags))
        for e in range(42):
            plot_ac = acf_cesm[v][e]
            ax.plot(lags,plot_ac,alpha=0.1,color=ac_colors[v],lw=2,label="",zorder=3)
            if not np.all(plot_ac==0):
                ncount += 1
                meanacf += plot_ac
        meanacf /= ncount
        ax.plot(lags,meanacf,color=ac_colors[v],lw=2,label=varnames_ac[v] +" (ens. mean)" )
        
    else:
        for e in range(42):
            plot_ac = ac_pt[v][e,:,kmonth]
            ax.plot(lags,plot_ac,alpha=0.1,color=ac_colors[v],lw=2,label="",zorder=3)
        ax.plot(lags,np.nanmean( ac_pt[v][:,:,kmonth],0),color=ac_colors[v],lw=2,label=varnames_ac[v] +" (ens. mean)" )

for s in range(3):
    ax.plot(lags,acs_out[s],label=sm_acs_name[s],ls='dashed',c=sm_acs_color[s],lw=2)
ax.legend()

#%% Compare the power spectra

nsmooth = 3
pct     = 0.05
opt     = 1
dt      = 3600*24*30
clvl    = 0.95

spec_output = scm.quick_spectrum(output_ts,nsmooth=nsmooth,pct=pct,opt=opt,return_dict=True)

for v in range(2):
    
    
    
    for e in range(42):
        invaracs
    
#cesm_specs  = scquick_spectrum(output_ts,nsmooth=nsmooth,pct=pct,opt=opt,return_dict=True)

#%% Plot it


fig,axs = plt.subplots(2,1)

for sim in range(2):
    
    ax       = axs[sim]
    plotvar  = spec_output['specs'][sim]
    plotfreq = spec_output['freqs'][sim]
    
    ax.plot(plotfreq*dt,plotvar/dt,label=sm_acs_name[sim],c=sm_acs_color[sim])
    
ax.legend()
    
    

#sm_T  = spec_output['specs'][0]
#sm_S1 =spec_output['specs'][0]


#%% Compare monthly variance

v    = 1
v_sm = 0


fig,ax = init_annplot()

plotvar_cesm= acvars_np[v][:,:,:].std(1) # [ens x mon]
for e in range(42):
    ax.plot(mons3,plotvar_cesm[e,:],alpha=0.2,color='gray')
ax.plot(mons3,plotvar_cesm.mean(0),color="k")

ax.plot(mons3,output_ts[v_sm].reshape(nyrs,12).std(0),color="blue")







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

# #%% Compute the precipitation flux term
# #sbar_mon # [ens mon z_t]
# prcp_total = np.zeros((nens,86,nmon))
# for p in prcps:
#     prcp_total += prcps_ds[p]
    
# # Compute the forcing term
# p_forcing = (sbar_mon.squeeze()[:40,None,:] * prcp_total[:40,:,:])/hpt[:40,None,:] # [ens x yr x mon]

#%% Compare the parameters I am using, to figure out what is fgoing on

# Compare Forcing
fig,ax = init_annplot()
ax.plot(mons3,Fpt,label="PiC Forcing")
ax.plot(mons3,Fori,label="Fori")
ax.set_ylabel("Qnet Forcing W/m2")
ax.legend()
#%% Compare Damping
fig,ax = init_annplot()
ax.plot(mons3,damppt,label="PiC Damping")
ax.plot(mons3,lbd_qnet,label="Htr Damping (Ens. Mean)")
for e in range(42):
    plotvar = dampings_pt[-1][e,:,0] # [damping][ens x mon x lag]
    ax.plot(mons3,plotvar,alpha=0.2,color="gray",label="",zorder=1)
               
ax.set_ylabel("Damping W/m2/degC")
ax.legend()
ax.set_title("Estimated Heat Flux Damping (Qnet) for CESM1")

#%% Compare MLD

fig,ax = init_annplot()
ax.plot(mons3,mldpt,label="PiC h")
ax.plot(mons3,h,label="Htr h (Ens. Mean)")
for e in range(40):
    plotvar = mld[e,:].reshape(int(1032/12),12).mean(0)/100 # [damping][ens x mon x lag]
    ax.plot(mons3,plotvar,alpha=0.2,color="gray",label="",zorder=1)
               
#ax.set_ylabel("Damping W/m2/degC")
ax.legend()
#ax.set_title("Estimated Heat Flux Damping (Qnet) for CESM1")


#%% Try Bokeh

import hvplot.xarray
import hvplot

var_in = np.array( dampings_pt )
coords = {"flux":flxs,
          "ens" :np.arange(1,43,1),
          "mon" :np.arange(1,13,1),
          "lag" :np.arange(1,4,1)
          }
da = xr.DataArray(var_in,coords=coords,dims=coords,name="Damping Estimates")

plot = da.sel(flux="qnet",lag=1).hvplot()
hvplot.show(plot)


# labels = LabelSet(x="month", y="Damping", text="symbol", y_offset=8,
#                   text_font_size="11px", text_color="#555555",
#                   source=source, text_align='center')
# p.hover.tooltips = [
#     ("name"         , "@name"),
#     ("symbol:"      , "@symbol"),
#     ("density"      , "@density"),
#     ("atomic weight", "@{atomic mass}"),
#     ("melting point", "@{melting point}"),
# ]



