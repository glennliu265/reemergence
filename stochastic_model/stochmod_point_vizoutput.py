#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize the output/inputs of stochmod_point.

Copied above section of the script.

Created on Thu Nov 30 10:50:50 2023

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
import yo_box as ybx

#%% User Edits

# Location
lonf           = 330
latf           = 50
locfn,loctitle = proc.make_locstring(lonf,latf)

datpath        = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/lon%s_lat%s/" % (lonf,latf)
figpath        = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20231201/"
proc.makedir(figpath)

flxs           = ["LHFLX","SHFLX","FLNS","FSNS","qnet"]
prcps          = ["PRECC","PRECL","PRECSC","PRECSL",]
varnames_ac    = ["SST","SSS"]

# Forcing from 
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
kprevpt  = np.array([2.52206314, 2.        , 0.        , 0.        , 0.        ,
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

#%% Load some information from [stochmod_point_dampingest]

Fprime_re   = np.load(datpath+"CESM1_htr_Fprime_forcing.npy")
lbda_re     = np.load(datpath+"CESM1_htr_lbda_reestimate.npz",allow_pickle=True)
qL_re       = np.load(datpath+"CESM1_htr_qL_forcing.npy")
lbds_re     = np.load(datpath+"CESM1_htr_lbds_reestimate.npz")
lbdts_re    = np.load(datpath+"CESM1_htr_lbdts_reestimate.npz")
vars_noenso = np.load(datpath+"CESM1_htr_vars_noenso.npz",allow_pickle=True)

#%% Load Tdexp Damping calculated by [calc_Td_decay.py]

outpath_sminput = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/"
savename        = "Td_Damping_HTR_ens01_13lagfit.npz"
ld_Td           = np.load(outpath_sminput+"Td_Damping_HTR_ens01_13lagfit.npz",allow_pickle=True)
lonf180         = lonf-360
klonTd,klatTd   = proc.find_latlon(lonf180,latf,ld_Td['lon'],ld_Td['lat'])
Tddamp          = ld_Td['Tddamp'][klonTd,klatTd,:]
print(np.exp(-Tddamp))


# ><><><><><><><><><><><><><><><><><><><><>< ><><><><><><><><><><><><><><><><><><><><><
#%% Parameter Loading Section
# ><><><><><><><><><><><><><><><><><><><><>< ><><><><><><><><><><><><><><><><><><><><><
# Get the fluxes
flxs_ds = {}
flxa_ds = {}
for f in range(5):
    vname = flxs[f]
    if vname == "qnet":
        ds_new = flxs_ds['FSNS'] - (flxs_ds['LHFLX'] + flxs_ds['SHFLX'] + flxs_ds['FLNS'])
        ds     = ds_new.rename(vname)
    else:
        ds = xr.open_dataset("%sCESM1_htr_%s.nc" % (datpath,vname))[vname].load() # [ens x time]

    # Compute the values
    flxs_ds[vname] = ds.copy()
    
    # Compute the anomaly
    dsa = preproc_pt(ds)
    flxa_ds[vname] = dsa.copy()

#%% Get the precip fields

prcps_ds  = {}
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

# <o><o><o><o><o><o><o><o><o><o><o><o><o><o>
#%% Check how the stdev(SST,SSS) each season looks like (across ensemble member)
# <o><o><o><o><o><o><o><o><o><o><o><o><o><o>
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

# <o><o><o><o><o><o><o><o><o><o><o><o><o><o>
#%% AC Plot
# <o><o><o><o><o><o><o><o><o><o><o><o><o><o>

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


# Salinity Feedback Damping
lbds_lhf      = np.nanmean(lbds_re['LHFLX'][:,:,0],0)  # [ens x mon x lag]


# Get SST and SSS with ENSO removed
ssta          = vars_noenso['SST'] # [ens x time]
sssa          = vars_noenso['SSS'] # [ens x time]

## Get seasonal values of F' ---------------------------------------------------



# Maybe for organization, let's collect the parameters here.



#%%
## Estimate Precipitation forcing
hvary = 0

if hvary:
    h_in  = h_emean
else:
    h_in = h_emean.mean()
ptot_mon   = np.nanstd(ptot,(0,1))
ptot_force = (ptot_mon * sbar_mon.mean(0).squeeze()) * dt / h_in


h          = h_in
evap_force = (qL_re.mean(0) / (rho*L*h) * dt * sbar)

# <o><o><o><o><o><o><o><o><o><o><o><o><o><o>
#%% Plot Precipitation Forcing
# <o><o><o><o><o><o><o><o><o><o><o><o><o><o>
fig,ax      = plt.subplots(1,1,figsize=(8,4))

ax.plot(mons3,ptot_force,label="P'")
ax.plot(mons3,evap_force,label="$q_L$'")
ax.set_ylabel("Precip. Forcing (psu/mon)")
ax.set_title(r"P'($\frac{ \overline{S} P'}{\rho h}$) and E' ($\frac{ \overline{S} q_L'}{\rho hL}$) Forcing @ %s" % (loctitle))
ax.set_xlim([0,11])
# savename = "%sCESM1_Precipitation_Flux_Scycle.png" % figpath
# plt.savefig(savename,dpi=150,bbox_inches='tight')
ax.legend()
ax.grid(True,ls='dotted')
savename = "%sStochmod_P_E_Forcing_hvary%i.png" % (figpath,hvary)
plt.savefig(savename,dpi=150,bbox_inches='tight')

# <o><o><o><o><o><o><o><o><o><o><o><o><o><o>
#%% Compare Damping
# <o><o><o><o><o><o><o><o><o><o><o><o><o><o>

fig,ax      = plt.subplots(1,1,figsize=(8,4))

ax.plot(mons3,damppt,label="CESM1-PiControl")
ax.plot(mons3,lbda_re['qnet'].mean(0)[:,0] * -1,label="CESM1-HTR")

ax.set_ylabel("Damping (W/m2/degC)")
ax.set_title(u"$\lambda _a$ ($Q_{net}$) Comparison @ %s" % (loctitle))
ax.set_xlim([0,11])
ax.legend()
ax.grid(True,ls='dotted')
savename = "%sStochmod_compare_damping.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')

# <o><o><o><o><o><o><o><o><o><o><o><o><o><o>
#%% Compare MLD
# <o><o><o><o><o><o><o><o><o><o><o><o><o><o>

#fig,ax = pl

fig,ax      = plt.subplots(1,1,figsize=(8,4))

ax.plot(mons3,mldpt,label="CESM1-PiControl")
ax.plot(mons3,h_emean,label="CESM1-HTR")
ax.set_ylabel("h (m)")
ax.set_title(u"MLD Comparison @ %s" % (loctitle))
ax.set_xlim([0,11])
ax.legend()
ax.grid(True,ls='dotted')
savename = "%sStochmod_compare_MLD.png" % (figpath,)
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
eta2   = np.random.normal(0,1,12*nyrs)

# Indicate if you have seasonal variation
svary     = True
hvary     = True # True to have seasonal h variation (in denominators)
fvary     = True # True to have seasonal forcing amplitude variation
dvary     = True # True to have seasonal damping variation
use_qL    = False # Force with just the latent heat component for temperature
samenoise = True
use_lbdts = True
expstr = "hvary%i_fvary%i_dvary%i_samenoise%i_qLT/" % (hvary,fvary,dvary,samenoise)
expdir = figpath+expstr
proc.makedir(expdir)

# Set some universal variables
# if svary:
#     h       = h_emean #mldpt#h_emean # [mon]
# else:
#     h       = h_emean.mean() * np.ones(12)#h_emean.mean() * np.ones(12)

# Set h in the denominator of the forcing + damping
h = h_emean
if hvary:
    h_denom = h_emean
else:
    h_denom = h_emean.mean() * np.ones(12)

#%% Do Unit Conversions (for temperature) ---------------------------------------

if use_qL:
    F_in = qL_re.mean(0)
    d_in = lbda_re['LHFLX'].mean(0)[:,0]
else:
    F_in = Fprime_re.mean(0)
    d_in = lbda_re['qnet'].mean(0)[:,0] * -1

if fvary:
    Fmag = F_in
else:
    Fmag = F_in.mean() * np.ones(12)


if dvary:
    hff    = d_in#lbda_re['qnet'].mean(0)[:,0] * -1#lbd_lhf# lbd_qnet
else:
    hff    = d_in.mean() * np.ones(12)#lbd_qnet.mean()*np.ones(12)

# Do conversions
lbd_a  = hff / (rho*cp*h_denom) * dt
alpha  = Fmag / (rho*cp*h_denom) * dt
F_T    = np.tile(alpha,nyrs) * eta

# First integrate temperature -------------------------------------------------
T      = scm.integrate_entrain(h[None,None,:],
                               kprev_mean[None,None,:],
                               lbd_a[None,None,:],
                               F_T[None,None,:],
                               T0=np.zeros((1,1)),
                               multFAC=True,
                               debug=False,
                               Td0=False)


Tdict = scm.integrate_entrain(h[None,None,:],
                               kprev_mean[None,None,:],
                               lbd_a[None,None,:],
                               F_T[None,None,:],
                               T0=np.zeros((1,1)),
                               multFAC=True,
                               debug=True,
                               Td0=False,return_dict=True)


# T_inputs =
#     {
#      "lbd_a" : lbd_a,
#      "alpha" : alpha,
     
#      }
# #%% Peak at the params

# fig,ax = init_annplot()
# ax.plot(mons3,lbd_a,label="Damping",color="orange")
# ax.plot(mons3,alpha,label="Forcing",color="limegreen")
# ax.set_ylabel("deg C / mon")
# ax.set_title("Stochastic Model Inputs (T')")
# ax.legend()

#%% Try above, but with Td Damping

Tdict_Tddamp = scm.integrate_entrain(h[None,None,:],
                               kprev_mean[None,None,:],
                               lbd_a[None,None,:],
                               F_T[None,None,:],
                               T0=np.zeros((1,1)),
                               multFAC=True,
                               debug=True,
                               Td0=False,return_dict=True,
                               Tdexp=Tddamp[None,None,:])

#%% Now try the version lambda_a * S' -----------------------------------------
# Note that this is essentially the same as above, but with weaker damping
# The forcing term should theoretically be reesimated but I didnt...


# lbd_lhf_cust = np.array([16.96781049, 13.30814792, 5, 5, 11.5776733 ,
#        10.75794417, 11.08759759, 12.94189574, 14.99567006, 17.29924383,
#        18.58622274, 18.9057527 ])

# h = np.array([137.26482 , 154.96883 , 150, 150,  100,
#         25,  23, 26 ,  30,  51.93153 ,
#         73.71345 , 102.56731 ])

lbd_lhf_cust = np.array([16.96781049, 13.30814792, 11, 11, 0,
        0, 0, 0, 0, 0,
        18.58622274, 18.9057527 ])

ptot_force_cust = np.array([0.0098616 , 0.00881426, 0.00973901, 0.01416393, 0.0270726 ,
       0.03760537, 0.03685692, 0.03333382, 0.03046539, 0.02448979,
       0.01853393, 0.01280091])

if samenoise:
    eta_in = eta
else:
    eta_in = eta2

if fvary:
    ptot_in = ptot_force    # Precipitation Forcing (psu/mon)[mon]
    qL_in   = qL_re.mean(0) # Evaporation Forcing (W/m2) [ens x mon]
else:
    ptot_in = ptot_force.mean() * np.ones(12)
    qL_in   = qL_re.mean(0).mean(0) * np.ones(12)

if dvary:
    hff    = lbda_re['LHFLX'].mean(0)[:,0] #lbd_lhf#_cust
else:
    hff    = lbda_re['LHFLX'].mean(0)[:,0].mean(0) * np.ones(12) #lbd_lhf.mean() * np.ones(12)

# Conversions and prep
lbd_a     = hff / (rho*cp*h_denom) * dt
alpha     = (ptot_in) + (qL_in) / (rho*L*h_denom) * dt * sbar
F         = np.tile(alpha,nyrs) * eta_in
F2        = np.tile(alpha,nyrs) * eta2

# Integrate for S1 (Salinity Method 1)
S1      = scm.integrate_entrain(h[None,None,:],
                               kprev_mean[None,None,:],
                               lbd_a[None,None,:],
                               F[None,None,:],
                               T0=np.zeros((1,1)),
                               multFAC=True,
                               debug=False,
                               Td0=False)

S1_dict= scm.integrate_entrain(h[None,None,:],
                               kprev_mean[None,None,:],
                               lbd_a[None,None,:],
                               F[None,None,:],
                               T0=np.zeros((1,1)),
                               multFAC=True,
                               debug=True,
                               Td0=False,
                               return_dict=True)

S1_dict_diffnoise= scm.integrate_entrain(h[None,None,:],
                               kprev_mean[None,None,:],
                               lbd_a[None,None,:],
                               F2[None,None,:],
                               T0=np.zeros((1,1)),
                               multFAC=True,
                               debug=True,
                               Td0=False,
                               return_dict=True)

#%% Now do the version where we use lbd_e*T' ----------------------------------

if fvary:
    ptot_in = ptot_force    # Precipitation Forcing (psu/mon)[mon]
    qL_in   = qL_re.mean(0) # Evaporation Forcing (W/m2) [ens x mon]
else:
    ptot_in = ptot_force.mean() * np.ones(12)
    qL_in   = qL_re.mean(0).mean(0) * np.ones(12)

if dvary:
    hff_ts = np.nanmean(lbdts_re['LHFLX'],0)[:,0]
    hff    = lbda_re['LHFLX'].mean(0)[:,0] #lbd_lhf#_cust
    hff_s  = lbds_lhf
else:
    hff    = lbda_re['LHFLX'].mean(0)[:,0].mean(0) * np.ones(12) #lbd_lhf.mean() * np.ones(12)
    hff_s  = lbds_lhf.mean(0) * np.ones(12)

lbd_a  = hff / (rho*cp*h_denom) * dt
lbd_e  = (sbar*cp*lbd_a)/ (L*(1+B))
lbd_s  = hff_s * sbar/ (rho*L*h_denom) * dt
add_F  = np.tile(lbd_e,nyrs) * np.roll(T[0,0,:],1,axis=0) *-1 #*10 * -1 * np.sign(eta_in)#0# lbd_e * T'
#lbd_a  = hff_ts / (rho*cp*h_denom) * dt#np.array([0.0,])#lbd_s#np.array([0.05,])
lbd_a    = np.array([0,],) #np.array([0.1,]) # using .1 gets damn close...
#lbd_a    = np.array([0.1,]) # using .1 gets damn close...
F        = np.tile(alpha,nyrs) * eta_in
F2        = np.tile(alpha,nyrs) * eta2


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

S2_dict = scm.integrate_entrain(h[None,None,:],
                               kprev_mean[None,None,:],
                               lbd_a[None,None,:],
                               F[None,None,:],
                               T0=np.zeros((1,1)),
                               multFAC=True,
                               debug=True,
                               Td0=False,
                               add_F=add_F[None,None,:],
                               return_dict=True)

S2_dict_diffnoise = scm.integrate_entrain(h[None,None,:],
                               kprev_mean[None,None,:],
                               lbd_a[None,None,:],
                               F2[None,None,:],
                               T0=np.zeros((1,1)),
                               multFAC=True,
                               debug=True,
                               Td0=False,
                               add_F=add_F[None,None,:],
                               return_dict=True)

t = np.arange(0,len(eta_in)) # Make a t variable for plotting

# <o><o><o><o><o><o><o><o><o><o><o><o><o><o>
#%% Quickly Examine the scatter between different variables
# <o><o><o><o><o><o><o><o><o><o><o><o><o><o>

fig,axs = plt.subplots(2,2,figsize=(12,10),constrained_layout=True)

# Plot the temperature anomalies versus the initial salinity anomalies
# with the feedback substitution
ax = axs[0,0]
sc = ax.scatter(T,S1,c=t,alpha=0.25)
ax.set_title("T vs. S1")
ax.set_xlabel("T")
ax.set_ylabel("S1")
ax.axhline([0],lw=0.75,c="k")
ax.axvline([0],lw=0.75,c="k")

# Plot the same as above but consider the SST-evaporation feedback
ax = axs[0,1]
ax.scatter(T,S2,c=t,alpha=0.25)
ax.set_title("T vs. S2")
ax.set_xlabel("T")
ax.set_ylabel("S2")
ax.axhline([0],lw=0.75,c="k")
ax.axvline([0],lw=0.75,c="k")

# Compare the F' and lambda_e terms to see if they are in covariance
ax = axs[1,0]
sc = ax.scatter(F,add_F,c=t,alpha=0.25)
ax.set_title("F' vs. $\lambda ^eT$")
ax.set_xlabel("F")
ax.set_ylabel("$\lambda ^eT$")
ax.axhline([0],lw=0.75,c="k")
ax.axvline([0],lw=0.75,c="k")

# Compare the differences between the forcing and damping terms
# check this to make sure it is alright
ax = axs[1,1]
sc = ax.scatter(S2_dict['forcing_term'],S2_dict['damping_term'],c=t,alpha=0.25)
ax.set_title("F' vs. $\lambda ^eT$")

# Look at the colorbar between the two
# information
cb  = fig.colorbar(sc,ax=axs.flatten(),orientation='horizontal',fraction=0.05,)
cb.set_label("Model Step")

#plt.scatter(T,S2)
# ><><><><><><><><><><><><><><><><><><><><>< ><><><><><><><><><><><><><><><><><><><><><
#%% Analysis Section
# ><><><><><><><><><><><><><><><><><><><><>< ><><><><><><><><><><><><><><><><><><><><><



#%% Calculate, plot and compare the ACF Calculate the autocorrelation --------

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
output_ts       = [T,S1,S2,S1_dict_diffnoise['T'],S2_dict_diffnoise['T'],Tdict_Tddamp['T']]
output_ts       = [ts.squeeze() for ts in output_ts]
acs_out,cfs_out = scm.calc_autocorr(output_ts,lags,kmonth,calc_conf=True,)
sm_acs_name     = ["T (SM)",
                   "S($\lambda_{LHF} S'$)","S ($\lambda_e T'$)",
                   "S($\lambda_{LHF} S'$) (diff noise)",
                   "S ($\lambda_e T'$) (diff noise)", 
                   "T (SM, Td Damp)",
                   ]
sm_acs_fn       = ["T","S1","S2","S1 (diffnoise)","S2 (diffnoise)","T (Td Damp)"]
sm_acs_color    = ["indianred","indigo","violet","cornflowerblue","pink","darkblue"]
sm_acs_ls       = ["dashed","dashed","dotted","solid","dashdot",'dashed']

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
    fig,ax   = plt.subplots(1,1,figsize=(12,4.5),constrained_layout=True)
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

for s in range(len(acs_out)):
    if s == 0:
        iv =0
    else:
        iv =1
    ax.plot(lags,acs_out[s],label=sm_acs_name[s],c=sm_acs_color[s],lw=2.5,ls=sm_acs_ls[s])
ax.legend(ncol=2,fontsize=14)
ax.set_ylim([-.25,1.25])

figname = "%sCESM1_v_SM_ACF_%s.png" % (expdir,locfn)
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% Focus on plot with Td' Damping

plotids = [0,5]
# Make the plot
if sameplot:
    fig,ax   = plt.subplots(1,1,figsize=(12,4.5),constrained_layout=True)
else:
    fig,axs   = plt.subplots(2,1,figsize=(8,8),constrained_layout=True)

v = 0
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

for ids in range(len(plotids)):
    s = plotids[ids]
    
    if s == 0:
        iv =0
    else:
        iv =1
    ax.plot(lags,acs_out[s],label=sm_acs_name[s],c=sm_acs_color[s],lw=2.5,ls=sm_acs_ls[s])
ax.legend(ncol=2,fontsize=14)
ax.set_ylim([-.25,1.25])

ax.plot(lags,cesmauto,color="k",label='CESM1 PiControl')
ax.plot(lags,oldintegration,color='cornflowerblue',label="Old Stochastic Model")

figname = "%sCESM1_v_SM_ACF_%s_Tddamp.png" % (expdir,locfn)
plt.savefig(figname,dpi=150,bbox_inches='tight')


#%% Plot the timeseries

fig,axs = plt.subplots(3,1,constrained_layout=True,figsize=(12,8))
for sim in range(5):
    
    ax  = axs[sim]
    
    lbl = "%s, $\sigma$=%f" % (sm_acs_name[sim],np.std(output_ts[sim]))
    ax.plot(output_ts[sim],label=lbl,c=sm_acs_color[sim])
    
    
    
    #ax.plot(proc.lp_butter(output_ts[sim],240,6),c="k")
    ax.legend()
    
    figname = "%sCESM1_v_SM_Timeseries_%s.png" % (expdir,locfn)
    #ax.set_xlim([0,720])
    #ax.set_ylim([-1,1])
    plt.savefig(figname,dpi=150,bbox_inches='tight')
    
    
#%% Plot the timeseries by season 

#%% Compare the power spectra

nsmooth = 25
pct     = 0.05
opt     = 1
dt      = 3600*24*30
clvl    = 0.95

spec_output      = scm.quick_spectrum(output_ts,nsmooth=nsmooth,pct=pct,opt=opt,return_dict=True)
#spec_output_cesm = 

#%% Plot it

plotbars = (100,50,20,10,5,1)
fig,axs = plt.subplots(3,1,constrained_layout=True,figsize=(12,8))

for sim in range(3):
    
    ax       = axs[sim]
    plotvar  = spec_output['specs'][sim]
    plotfreq = spec_output['freqs'][sim]
    ax.plot(plotfreq*dt,plotvar/dt,label=sm_acs_name[sim],c=sm_acs_color[sim])
    ax.set_xlim([1/(12*100),1/(12)])
    
    
    for pb in plotbars:
        ax.axvline([pb],ls='dotted',color="k")
    
    ax.set_xlabel("Cycles/Mon")
    ax.legend()
    
    

#sm_T  = spec_output['specs'][0]
#sm_S1 =spec_output['specs'][0]


#%% Compare monthly variance

for sim in range(3):
    fig,ax = init_annplot()
    if sim == 0:
        v = 0
    else:
        v = 1
    
    plotvar_cesm= acvars_np[v][:,:,:].std(1) # [ens x mon]
    for e in range(42):
        ax.plot(mons3,plotvar_cesm[e,:],alpha=0.2,color='gray')
    ax.plot(mons3,plotvar_cesm.mean(0),color="k")
    ax.plot(mons3,output_ts[sim].reshape(nyrs,12).std(0),color="blue")
    ax.set_title("CESM1 vs. Stochastic model (%s)\n %s" % (sm_acs_name[sim],expstr))
    
    figname = "%sCESM1_v_SM_variance_%s_%s_%s.png" % (expdir,locfn,expstr,sm_acs_fn[sim])
    plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% Look at scatter relationship between T' and S

fig,axs = plt.subplots(1,2,figsize=(12,8),constrained_layout=True,sharey=True)

t = np.arange(0,len(T.squeeze()))

ax = axs[0]
sc = ax.scatter(T.squeeze(),S1.squeeze(),c=t,cmap='jet',alpha=0.25)
ax.set_ylabel(sm_acs_name[1])
ax.set_xlabel("T")

ax = axs[1]
sc = ax.scatter(add_F.squeeze(),S2,c=t,cmap='jet',alpha=0.25)
ax.set_xlabel("T")
ax.set_ylabel(sm_acs_name[2])

cb = fig.colorbar(sc,ax=axs.flatten(),fraction=0.05,pad=0.05,orientation='horizontal')
cb.set_label("Simulation Time (Months)")

figname = "%sCESM1_v_SM_scatter_%s_%s.png" % (figpath,locfn,expstr)
plt.savefig(figname,dpi=150,bbox_inches='tight')
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


#%% Plot Damping

lbd_s_svary  = hff_s * sbar/ (rho*L*h_emean) * dt
lbd_s_hconst = hff_s * sbar/ (rho*L*h_emean.mean()) * dt
lbd_s_dconst = hff_s.mean() * sbar/ (rho*L*h_emean) * dt
lbd_s_const  = hff_s.mean() * sbar/ (rho*L*h_emean.mean()) * dt

fig,ax = init_annplot(figsize=(8,4))
ax.plot(mons3,lbd_s_svary,label="All Vary")
ax.plot(mons3,lbd_s_hconst,label="H Const (%.2f [m])" % (h_emean.mean()))
ax.plot(mons3,lbd_s_dconst,label="$\lambda_S$ Const (%.2f [W/m2/psu])" % (hff_s.mean()))
ax.axhline([lbd_s_const],label="All Const ( %f [psu/mon])" % lbd_s_const)
ax.axhline(0.05,color="k",label="0.05 psu/mon")
ax.legend(fontsize=12,ncol=2)
ax.set_ylim([0,0.11])

figname = "%sLbd_S_Comparison_%s.png" % (figpath,locfn)
plt.savefig(figname,dpi=150,bbox_inches='tight')


#%% Plot Lag Correlation of SST and SSS
ssta_clean = ssta.copy()
sssa_clean = sssa.copy()
ssta_clean[np.isnan(ssta)] = 0
sssa_clean[np.isnan(sssa)] = 0


output        =  proc.calc_lag_covar_ann(ssta_clean.T,sssa_clean.T,lags,dim=0,detrendopt=1)
ts_corr       = output[0]

output1       =  proc.calc_lag_covar_ann(sssa_clean.T,ssta_clean.T,lags,dim=0,detrendopt=1)
ts_corr_slead = output1[0]

# Manually do this

e     = 0
p     = 0.05
tails = 2

# Lead for all ensemble member
corrs_tlead_all = []
corrs_slead_all = []
ts_dof     = []
ts_rhocrit = []
for e in range(42):
    ntime= ssta.shape[1]
    
    corrs_tlead = []
    corrs_slead = []
    dofs     = []
    rhocrits = []
    # Loop for each Lag time
    for l in range(len(lags)):
        ## SST Leads
        lag = lags[l]
        ts0 = ssta[e,:(996-lag)]
        ts1 = sssa[e,lag:]
        ct  = np.corrcoef(ts0,ts1)[0,1]
        corrs_tlead.append(ct)
        
        # SSS leads
        ts0 = sssa[e,:(996-lag)]
        ts1 = ssta[e,lag:]
        cs  = np.corrcoef(ts0,ts1)[0,1]
        corrs_slead.append(cs)
        
    # Compute DOFs
    dof_eff = proc.calc_dof(ssta[e,:],ts1=sssa[e,:])
    rhocrit = proc.ttest_rho(p,tails,dof_eff)
    ts_dof.append(dof_eff)
    ts_rhocrit.append(rhocrit)
    
    
    
    corrs_tlead_all.append(np.array(corrs_tlead))
    corrs_slead_all.append(np.array(corrs_slead))
    
#%% Plot Lead Lag of T and S


neg_lags= np.flip(lags)*-1

xtks = np.arange(-40,41,5)

fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(8,5.5))

ax = viz.add_ticks(ax)

slead_mean = np.zeros(len(lags))
tlead_mean = np.zeros(len(lags))


scount = 0
tcount = 0
for e in range(42):
    
    corrs_slead = corrs_slead_all[e]
    corrs_tlead = corrs_tlead_all[e]
    
    ax.plot(neg_lags,np.flip(np.array(corrs_slead)),alpha=0.20,color="blue")#label="SSS Leads")
    ax.plot(lags,corrs_tlead,alpha=0.20,color='orange')#label="SST Leads")
    
    if not np.any(np.isnan(corrs_slead)):
        slead_mean += np.flip(np.array(corrs_slead))
        scount += 1
    if not np.any(np.isnan(corrs_slead)):
        tlead_mean += corrs_tlead
        tcount += 1
    
ax.plot(neg_lags,slead_mean/scount,label="SSS Leads",color="darkblue")
ax.plot(lags,tlead_mean/tcount,label="SST Leads",color="darkorange")
ax.axhline([0],color="k",lw=0.75)
ax.axvline([0],color="k",lw=0.75)

ax.axhline(np.nanmax(ts_rhocrit),color='gray',ls='dashed',
           lw=0.75,label="r = %f" % (np.nanmax(ts_rhocrit)))

ax.set_xticks(xtks)
ax.set_xlim([-36,36])
ax.set_ylim([-.25,1,])
ax.set_xlabel("SSS Lag (Months)")
ax.set_ylabel("Correlation")

#ax.plot(neg_lags,np.flip(ts_corr_slead[:,e]),label="SSS Leads (func)")
#ax.plot(lags,np.flip(ts_corr[:,e]),label="SST Leads (func)")
ax.legend()

figname = "%sTS_LeadLag_Comparison_%s.png" % (figpath,locfn)
plt.savefig(figname,dpi=150,bbox_inches='tight')
#lags_new = np.flip(lags) * -1
#ts_corrneg,_ = proc.calc_lag_covar_ann(ssta_clean.T,sssa_clean.T,lags_new,dim=0,detrendopt=1)
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

#%% Compare the relative amplitudes of the forcing terms



keys = list(Tdict.keys())
nkeys = len(keys)
print(Tdict.keys())
print(S2_dict.keys())




termnames              = ["T","damping_term","forcing_term","entrain_term","Td"]
termnames_S            = ["S","damping_term","forcing_term","entrain_term",'Sd']
termcolors             = ["k","r","magenta","cornflowerblue","cyan"]

indict = Tdict
[print("var(%s) = %f" % (keys[k],np.var(indict[keys[k]]))) for k in range(len(keys))]

term_vars = []
indicts   = [Tdict,S1_dict,S2_dict]
for v in range(3):
    
    indict       = indicts[v]
    
    print("\n")
    [print("var(%s) = %f" % (keys[k],np.var(indict[keys[k]]))) for k in range(len(keys))]
    
    term_var_mon = np.zeros((12,len(termnames)))
    
    for k in range(len(termnames)):
        
        key = keys[k]
        ts  = indict[key].squeeze()
        _,ts = proc.calc_clim(ts,0,returnts=1) # [yr, mon]
        print("var for %s is %s" % (key,np.var(ts,axis=0)))
        for im in range(12):
            term_var_mon[im,k] = np.var(ts[:,im]) 
    term_vars.append(term_var_mon.copy())
    
    

# Compute the Monthly variance for add_F
_,add_F_mon = proc.calc_clim(add_F,dim=0,returnts=True)
var_addF    = np.var(add_F_mon,axis=0)


# Extract the FAC terms
FAC_T = Tdict['FAC']
FAC_S = S2_dict['FAC']

#%%

fig,axs = plt.subplots(3,1,figsize=(12,10))

#ax.bar(termnames,term_vars[0])

# Plot Variables for Temperature
v = 0
ax = axs[v]
for tv in range(len(termnames)):
    
    
    if tv == 2 or tv == 3:
        mult = (FAC_T.squeeze()**2)
    else:
        mult = 1
    
    plotvar = term_vars[v][:,tv] * mult
    ax.plot(mons3,plotvar,label=termnames[tv],c=termcolors[tv])
ax.legend(fontsize=12)
ax.set_ylabel("var(%s)" % varnames_ac[v])

# Plot for Salinity with Damping Substitution
v = 1
ax = axs[v]
for tv in range(len(termnames)):
    
    if tv == 2 or tv == 3:
        mult = (FAC_S.squeeze()**2)
    else:
        mult = 1
    plotvar = term_vars[v][:,tv] * mult
    ax.plot(mons3,plotvar,label=termnames_S[tv],c=termcolors[tv])
ax.legend(fontsize=12)
ax.set_ylabel("var(%s)" % varnames_ac[v])

# Plot Salinity with SST-Evaporation Feedback
v = 2
ax = axs[v]
for tv in range(len(termnames)):
    
    if tv == 2 or tv == 3:
        mult = 1#(FAC_S.squeeze()**2)
    else:
        mult = 1
    
    if tv == 2: # Forcing Term
        ls='dotted'
        plotvar = (term_vars[v][:,tv]-var_addF) * mult
        lbl="E-P Forcing"
    else:
        plotvar = term_vars[v][:,tv] * mult 
        lbl=termnames_S[tv]
        ls='solid'
    print(tv)
    print(plotvar)
    ax.plot(mons3,plotvar,label=lbl,c=termcolors[tv],ls=ls)
    ax.set_ylim([0,.003])
    
ax.plot(mons3,(FAC_S.squeeze()**2) * var_addF,color='limegreen',label="$\lambda_e$ T",ls='dashed')
ax.legend(fontsize=12)
ax.set_ylabel("var(%s)" % varnames_ac[1])
#ax.set_ylim([0,2e-03])

#%% Plot the coherence for all ensemble members

# pct     = 0.10
# opt     = 1
# nsmooth = 10


# for e in range(42):

# CP,QP,freq,dof,r1_x,r1_y,PX,PY = ybx.yo_cospec(acvars_np[0][e].flatten(),
#                                                acvars_np[1][e].flatten(),
#                                                opt,nsmooth,pct,
#                                                debug=False,verbose=False,return_auto=True)

# coherence_sq = CP**2 / (PX * PY)

#%% Plot the coherence

#xtks   = np.array([100*12,75*12,50*12,25*12,10*12,5*12])
#dtplot = 365*24*3600
#xfreq  = [1/(x)  for x in xtks]

xtks = np.arange(0,.12,0.02)
xtk_lbls = [1/(x)  for x in xtks]

fig,axs = plt.subplots(3,1,figsize=(12,10),constrained_layout=True)

for a in range(3):
    ax = axs[a]
    
    if a == 0:
        plotvar = PX
        lbl     = "SST"
        ylbl    = "$\degree C^2 \, cycles^{-1} \, mon^{-1}$"
    elif a == 1:
        plotvar = PY
        lbl     = "SSS"
        ylbl    = "$psu^2 \, cycles^{-1} \, mon^{-1}$"
    elif a == 2:
        plotvar = coherence_sq
        lbl     = "Coherence"
        ylbl    = "$psu \, \degree C \, cycles^{-1} \, mon^{-1}$"
    
    ax.plot(freq,plotvar,label="")
    
    
    
    if a == 2:
        ax.set_xlabel("cycles/mon")
    ax.set_title(lbl)
    #ax.set_xticks(xfreq)

    ax.axvline([1/12],color="k",ls='dashed',label="Annual")
    ax.axvline([1/60],color="k",ls='dashed',label="5-yr")
    ax.axvline([1/120],color="k",ls='dashed',label="Decadal")
    ax.axvline([1/1200],color="k",ls='dashed',label="Cent.")
    if a == 0:
        ax.legend(fontsize=16,ncols=2)
    ax.set_ylabel(ylbl)
    ax.set_xticks(xtks)
    ax.set_xlim([0,0.1])
    ax.grid(True,ls='dotted')
    
    #ax2 = ax.twiny()
    #ax2.set_xticks(xtks,labels=xtk_lbls)
    #ax2.set_xlabel("")
    
figname = "%sTS_Coherence_%s_ens%02i_nsmooth%i.png" % (figpath,locfn,e+1,nsmooth)
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% Examine the ratio of standard deviation of different terms

SST_cesm = vars_noenso['SST']
SSS_cesm = vars_noenso['SSS']
LHFLX    = vars_noenso['LHFLX']
QNET     = vars_noenso['qnet']

std_sst = np.nanstd(SST_cesm,1)
std_sss = np.nanstd(SSS_cesm,1)
std_lhf = np.nanstd(LHFLX,1)
std_net = np.nanstd(QNET,1)

std_pre = np.nanstd(ptot.reshape(42,86*12),1)

lhf_psu = ( sbar * std_lhf ) / (rho * h_emean.mean() * L  ) * dt
lhf_deg = (        std_lhf ) / (rho * h_emean.mean() * cp ) * dt

pre_psu = ( sbar * std_pre   / h_emean.mean() )             * dt
net_deg = (        std_net ) / (rho * h_emean.mean() * cp ) * dt

ratio_sss     = lhf_psu/std_sss
ratio_sst     = lhf_deg/std_sst
ratio_sss_pre = pre_psu/std_sss
ratio_sst_net = net_deg/std_sst

print("For T' ($Q_L$) : Mean: %.3f, Min to Max : [%.3f, %.3f]" % (ratio_sst.mean(),ratio_sst.min(),ratio_sst.max()))
print("For S' ($Q_L$) : Mean: %.3f, Min to Max : [%.3f, %.3f]" % (ratio_sss.mean(),ratio_sss.min(),ratio_sss.max()))
print("For S' (PREC)  : Mean: %.3f, Min to Max : [%.3f, %.3f]" % (ratio_sss_pre.mean(),ratio_sss_pre.min(),ratio_sss_pre.max()))
print("For T' (QNET)  : Mean: %.3f, Min to Max : [%.3f, %.3f]" % (ratio_sst_net.mean(),ratio_sst_net.min(),ratio_sst_net.max()))

#%% Check the 



#%% Check the monthly covariances of each of the component terms

fig,ax = plt.subplots(1,1)
ax.pcolormesh(lon,lat)

#%%

v = 2

indict = indicts[v]
fig,axs = plt.subplots(len(termnames),1,figsize=(12,12))
for tv in range(len(termnames)):
    ax = axs[tv]
    ax.plot(indict[termnames[tv]].squeeze(),label="%s, var=%f" % (termnames[tv],np.var(indict[termnames[tv]].squeeze())))
    ax.legend(fontsize=12)

#%%

# for v in range(2):
    
#     ax = axs[v]
    
#     for tv in range(len(termnames)):
#         ax.plot(mons3,term_vars[v][:,tv],label=termnames[tv],c=termcolors[tv])
#     if v == 0:
#         ax.legend(fontsize=12)
    
#     ax.set_ylabel("var(%s)" % varnames_ac[v])
#     #ax.set_ylim([0,0.04])
    
    
    
    


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



