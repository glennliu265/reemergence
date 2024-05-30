#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize Atmospheric Persistence

Examine if there is additional persistence in atmospheric variables such as Qnet,
Wind Modulus, and Stochastic Heat Flux Forcing that might explain increased persistence
in SST.
Works with output from [pointwise_crosscorrelation], or ACFs computed over the North Atlatnic
for the CESM1 52-member Large Ensemble Historical Period (1920-2005)


Created on Tue Apr  2 14:17:47 2024

@author: gliu
"""

from amv import proc, viz
import scipy.signal as sg
import yo_box as ybx
import amv.loaders as dl
import scm
import reemergence_params as rparams
import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import os
from tqdm import tqdm

# ----------------------------------
# %% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "Astraeus"

# First Load the Parameter File
sys.path.append("../")

# Paths and Load Modules
pathdict = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])

# Set needed paths
figpath = pathdict['figpath']
proc.makedir(figpath)
input_path = pathdict['input_path']
output_path = pathdict['output_path']
procpath = pathdict['procpath']

# %% User Edits

vnames = ["SST",
          "qnet",
          "Fprime",
          "Umod",
          "U",
          "V",
          "SLP",
          ]

vnames_long = [
    "$SST$",
    "$Q_{net}$",
    "Stochastic Heat Flux",
    "Wind Modulus",
    "Eastward Wind",
    "Northward Wind",
    "Sea Level Pressure"
]

vunits = [
    "$\degree C$",
    "$W m^{-2}$",
    "$W m^{-2}$",
    "m $s^{-1}$",
    "m/s",
    "m/s",
    "hPa"
]

vcolors = ["k",
           "royalblue",
           "violet",
           "limegreen",
           "goldenrod",
           "yellow",
           "cyan"]

vmarkers = ["o",
            "x",
            "+",
            "d",
            "x",
            "+",
            "^"]


ptpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/lon330_lat50/"

# %% Load the variables (basinwide ACFs computed throguh pointwise_crosscorrelation script)

nvars = len(vnames)
ds_all = []

for vv in tqdm(range(nvars)):

    vname = vnames[vv]

    if vname == "Fprime":  # Load data that has been processed by pointwise_crosscorrelation
        nc = procpath+"CESM1_1920to2005_%sACF_nomasklag1_nroll0_lag00to60_ALL_ensALL.nc" % vname
    else:
        nc = procpath+"CESM1_1920to2005_%sACF_lag00to60_ALL_ensALL.nc" % vname

    # [Ens x Lon x Lat x Mon x Thres x Lag]
    ds_all.append(xr.open_dataset(nc).acf.load())


# %%

def preprocess_ds(ds):
    # Remove mean seasonal cycle
    dsa = proc.xrdeseason(ds)  # Remove the seasonal cycle
    dsa = dsa - dsa.mean('ens')  # Remove the ensemble mean
    return dsa



def point_spectra(ts, nsmooth=1, opt=1, dt=None, clvl=[.95], pct=0.1):

    if dt is None:  # Divides to return output in 1/sec
        dt = 3600*24*30
    sps = ybx.yo_spec(ts, opt, nsmooth, pct, debug=False, verbose=False)

    P, freq, dof, r1 = sps
    coords = dict(freq=freq/dt)
    da_out = xr.DataArray(P*dt, coords=coords, dims=coords, name="spectra")
    return da_out


def get_freqdim(ts, dt=None, opt=1, nsmooth=1, pct=0.10, verbose=False, debug=False):
    if dt is None:
        dt = 3600*24*30
    sps = ybx.yo_spec(ts, opt, nsmooth, pct, debug=False, verbose=verbose)
    return sps[1]/dt




# %% Plot variables at a point

lonf = -30
latf = 50
dspt = [proc.selpt_ds(ds, lonf, latf)
        for ds in ds_all]  # [Ens x Mon x Thres x Lag]
locfn1, loctitle1 = proc.make_locstring(lonf, latf, lon360=True)


lonf2 = -58
latf2 = 45
locfn2,loctitle2 = proc.make_locstring(lonf2, latf2, lon360=True)

# %% Load data for the point (output from get_atm_vars.py)
locfns = [locfn1,locfn2]

dsatm_all = []
for ii in range(2):
    locfn = locfns[ii]
    
    ptpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/%s/" % locfn
    ptnc = "CESM1_HTR_FULL_NAtl_AtmoVar_%s.nc" % locfn
    dsatm = xr.open_dataset(ptpath + ptnc).load()
    
    dsatm_all.append(dsatm)
    
    
#%% Also load Fprime from old file


ptpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/%s/" % locfn1
ptnc     = "CESM1_HTR_FULL_NAtl_AtmoVar_%s_old.nc" % locfn1
dsatmold = xr.open_dataset(ptpath + ptnc).load()


# %%

# Plotting Parameters
ds = ds_all[0]
mpl.rcParams['font.family'] = 'JetBrains Mono'
bboxplot = [-80, 0, 20, 65]
proj = ccrs.PlateCarree()
lon = ds.lon.values
lat = ds.lat.values
mons3 = proc.get_monstr()

# %% Plot the ACFs for different atmospheric variables (works with T2 computed from crosscorrelation script)

# dspt = dsatm_all[0]

xtks = np.arange(0, 61, 6)
lags = np.arange(0, 61, 1)

for kmonth in range(12):

    title = "%s ACF @ %s" % (mons3[kmonth], loctitle)

    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(12, 4))
    ax, _   = viz.init_acplot(kmonth, xtks, lags, ax=ax, title=title)
    
    for vv in range(nvars):

        plotvar = dspt[vv].isel(mons=kmonth).squeeze()
        mu = plotvar.mean('ens')
        sigma = plotvar.std('ens')

        ax.plot(
            lags, mu, c=vcolors[vv], marker=vmarkers[vv], lw=2.5, label=vnames_long[vv])
        ax.fill_between(lags, mu-sigma, mu+sigma,
                        color=vcolors[vv], alpha=0.1, label="")

    ax.legend()
    ax.axhline([0], ls='solid', lw=0.75, color="k")

    savename = "%sAtmo_ACFs_%s_mon%02i.png" % (figpath, locfn, kmonth+1)
    plt.savefig(savename, dpi=150, bbox_inches='tight', transparent=True)

# %% Take the data and compute the spectra (usuing xarray ufuncs) (old single-point comparison)

ts = dsatm['SST'].isel(ens=0).values


opt = 1
nsmooth = 10
dt = None
clvl = [.95,]




specexp = []
acfexp  = []
for ii in range(3):
    
    
    
    
    specatm = []
    for vv in tqdm(range(nvars)):
    
        # Preprocess
        tsens = dsatm[vnames[vv]]
        tsens = preprocess_ds(tsens)
        
        # Compute Spectra
        specens = xr.apply_ufunc(
            point_spectra,  # Pass the function
            tsens,  # The inputs in order that is expected
            # Which dimensions to operate over for each argument...
            input_core_dims=[['time'],],
            output_core_dims=[['freq'],],  # Output Dimension
            exclude_dims=set(("freq",)),
            vectorize=True,  # True to loop over non-core dims
        )
    
        # Need to Reassign Freq as this dimension is not recorded
        ts1 = tsens.isel(ens=0).values
        freq = get_freqdim(ts1)
        specens['freq'] = freq
    
        specatm.append(specens)
    
    specexp.append(specatm)
    acfexp.append(acfatm)

#%% Read out to numpy arrays and redo spectra calculation
nsmooth = 1
pct     = 0.0
# taper

nens = 42
dtin = 3600*24*365
specvars = []
for vv in range(nvars):

    tsens = dsatm[vnames[vv]]
    tsens = preprocess_ds(tsens)
    tsens = tsens.groupby('time.year').mean('time')  # Take Annual Mean
    tsens = [tsens.isel(ens=e).values for e in range(nens)]
    specout = scm.quick_spectrum(tsens, nsmooth, pct, dt=dtin)
    specout = [np.array(s) for s in specout]

    specvars.append(specout)
        
    


# %% Try plotting the periodogram


pgrams = np.zeros((nvars, nens,))

for vv in range(nvars):

    tsens = dsatm[vnames[vv]]
    tsens = preprocess_ds(tsens)
    tsens = tsens.groupby('time.year').mean('time')  # Take Annual Mean
    tsens = [tsens.isel(ens=e).values for e in range(nens)]

    for e in range(nens):
        ff, ss = sg.periodogram(tsens[e], scaling="density")#scaling='spectrum')

        if (vv == 0) and (e == 0):
            nfreqs = len(ff)
            pgbyvar = np.zeros((nvars, nens, nfreqs))
            ffbyvar = np.zeros((nvars, nens, nfreqs))

        pgbyvar[vv, e, :] = ss.copy()
        ffbyvar[vv, e, :] = ff.copy()


# %% Plot the spectra using ufuncs


# xpers = [100, 50, 25, 10, 5, 2.5, 1]
# xtks = np.array([1/(t) for t in xpers])

# fig, axs = plt.subplots(nvars, 1, figsize=(12, 10))

# for vv in range(nvars):

#     ax = axs[vv]
#     plotvar = specatm[vv]
#     mu = plotvar.mean('ens')

#     ax.plot(mu.freq*dtplot, mu/dtplot, label=vnames_long[vv], c=vcolors[vv],)
#     ax.legend()

#     ax.set_xticks(xtks, labels=xpers)
#     ax.set_xlim([xtks[0], xtks[-1]])

#     # ax.axvline([1/(3600*24*365)])

# plt.suptitle("")

# %% Plot the spectra (manual pathway)
plot_perio = False
plot_log   = False
plot_ens   = True

dtplot = 3600*24*365  # Annual
xpers = [100, 50,25, 20, 15,10, 5, 2]
xtks = np.array([1/(t) for t in xpers])


fig, axs = plt.subplots(nvars, 1, figsize=(12, 10))

for vv in range(nvars):
    ax = axs[vv]

    svarsin = specvars[vv]
    P, freq, CCs, dof, r1 = svarsin

    # Convert units
    freq = freq[0, :] * dtplot
    P = P / dtplot
    Cbase = CCs.mean(0)[:, 0]/dtplot
    Cupbound = CCs.mean(0)[:, 1]/dtplot

    # Plot Axis
    mu = P.mean(0)
    sigma = P.std(0)
    
    # Plot Spectra
    if plot_log:
        ax.loglog(freq, mu, c=vcolors[vv], lw=2.5,
                label=vnames_long[vv], marker=vmarkers[vv])
        ax.fill_between(freq, mu-sigma, mu+sigma,
                        color=vcolors[vv], alpha=0.2, label="")
    else:
        ax.plot(freq, mu, c=vcolors[vv], lw=2.5,
                label=vnames_long[vv], marker=vmarkers[vv])
        ax.fill_between(freq, mu-sigma, mu+sigma,
                        color=vcolors[vv], alpha=0.2, label="")
    
    if plot_ens:
        for e in range(nens):
            plotens = P[e,:]
            if e == 0:
                lbl = "Indv. Ens"
            else:
                lbl = ""
            ax.plot(freq, plotens, c=vcolors[vv], lw=1,
                    label=lbl,alpha=0.08,zorder=-1)
    
    # Plot CCs
    ax.plot(freq, Cbase, color="crimson", ls='solid', lw=1.2, label="Red Noise")
    ax.plot(freq, Cupbound, color="crimson", ls="dotted",
            lw=1.2, label="95% Confidence")
    
    # Plot Periodogram
    if plot_perio:
        pgplot = pgbyvar[vv,:,:].mean(0) #* dtplot
        ffplot = ffbyvar[vv,0,:]
        ax.plot(ffplot,pgplot,c=vcolors[vv],lw=1.5,ls='dashed',label="Periodogram",marker=vmarkers[vv])

    # Set Labels and Ticks
    ax.set_ylabel("(%s)$^2$/cpy" % vunits[vv])
    ax.legend()
    ax.set_xticks(xtks, labels=xpers)
    ax.axvline([1/86],c="k",lw=.75)
    ax.axvline([1/2],c="k",lw=.75)
    ax.set_xlim([xtks[0], xtks[-1]])
    if vv == nvars-1:
        ax.set_xlabel("Period (Years)")
plt.suptitle("Power Spectra for Atmospheric Variables\n nsmooth=%i, taper=%.2f" % (nsmooth,pct),fontsize=24)


savename = "%sAtmo_Spectra_%s_Annual.png" % (figpath, locfn)
plt.savefig(savename, dpi=150, bbox_inches='tight', transparent=True)
    


#%% Functions to:
#% Compute the White Noise Spectra + CFs (move this to another script eventually)

import scipy as sp
def wnspectra_z(sigma,N,clvl =[.95,],tails=2,
                s=1,):
    # Generate confidence intervals above a white noise null hypothesis.
    # where [sigma] is the standard deviation of the timeseries of length [N].
    #     Use the formula: cf = sigma + z * s/sqrt(N)
    # s is 1 (assume errors of white noise are normally distributed). Z is obtained
    # using the confidence levels specified in clvl
    # if tails == 2:
    #     clvl = 1 - (1 - np.array(clvl))/2
    z     = sp.stats.norm.ppf(clvl)
    cfout = sigma + z * s / np.sqrt(N)
    return cfout

def wnspectra_mc(sigma,n,clvl=[.95,],mciter=10000):
    # Generate confidence intervals above a white noise null hypothesis use monte carlo
    # where [sigma] is the standard deviation of the timeseries of length [N].
    # Generates [mciter] number of white noise timeseries of the same length...
    sigmas = []
    for mc in range(mciter):
        wn = np.random.normal(0,sigma,n)
        sigmas.append(wn.std())
    cfout = np.quantile(sigmas,clvl)
    return cfout

def calc_wnspectra(ts,dim,freqs,clvl=[.95],mc=False):
    sigma  = np.nanvar(ts,axis=dim)
    N      = ts.shape[dim]
    if mc:
        cfout = wnspectra_mc(sigma,N,clvl=clvl)
    else:
        cfout = wnspectra_z(sigma,N,clvl=clvl)
    cfsall = []
    df     = freqs[-1] - freqs[0]
    nfreqs = len(freqs)
    cfsall.append(np.ones(nfreqs) * sigma/df ) # Base Level (distribute variance across frequencies)
    for sig in cfout:
        cfsall.append(np.ones(nfreqs) * sig/df ) # Add each level
    return cfsall

#%% 2024.04.10 Update: Compare Fprime Removal
# Settings
# compare_name   = 'Fprime_Test'
# vname          = 'Fprime'

compare_name = None # set to Fprime_Test to compare Fprime fix
calc_acf     = False
for iv in range(nvars):
    # if iv == 0:
    #     continue
    
    vname =vnames[iv]
    if compare_name is None:
        compare_name = vname
    
    expcols        = ["red","royalblue","violet"]
    print(vname)
    
    # Compare things for a single variable
    expnames       = [loctitle1,loctitle2]
    if compare_name == "Fprime_test":
        in_ts          = [dsatm_all[0][vname],dsatm_all[1][vname],dsatmold[vname]] # [Ens x Time]
        expnames       = expnames+[loctitle1 +" (F' Uncorrected)",]
    else:
        in_ts          = [ds[vname] for ds in dsatm_all]
    nexps          = len(expnames)
    
    # Get timeseries and take annual average
    tsa            = [preprocess_ds(ts) for ts in in_ts] # Detrend and Deseason
    tsann          = [ts.groupby('time.year').mean('time') for ts in tsa]  # Take Annual Mean
    
    
    # Calculate Spectra (copy from above section)
    nens     = 42
    dtin     = 3600*24*365
    pct      = 0.10
    nsmooth  = 5
    clvls    = [.95,]
    lags     = np.arange(37)
    specexp  = []
    wnexp    = []
    acfexp   = []
    for ex in range(nexps):
        
        # Compute Spectra
        tsens   = tsann[ex]
        tsens   = [tsens.isel(ens=e).values for e in range(nens)]
        specout = scm.quick_spectrum(tsens, nsmooth, pct, dt=dtin,make_arr=True,return_dict=True)
        specexp.append(specout)
        
        # Compute White Noise
        freqs   = specout['freqs'][0,:]
        wnspecs = np.array([calc_wnspectra(ts,0,freqs,clvl=clvl,mc=False) for ts in tsens]) # [Ens x CF x FREQ]
        wnspecs = wnspecs.transpose(0,2,1) # Ens x Freq x Clvl
        wnexp.append(wnspecs)
        
        # Compute ACF
        if calc_acf:
            tsensmon = tsa[ex]
            tsensmon = [tsensmon.isel(ens=e).values for e in range(nens)]
            tsmet    = np.array(scm.compute_sm_metrics(tsensmon,lags=lags)['acfs']) # [month,lag]
            acfexp.append(tsmet)
        
        
    
    specstr = "nsmooth%03i_taper%03i" % (nsmooth,pct*100)
    
    #% Compare the spectra ---------------------------------------
    
    
    
    plot_log   = True
    plot_ens   = False
    if vname == "SST":
        use_redn=True
    else:
        use_redn= False
    
    dtplot     = 3600*24*365  # Annual
    xpers      = [100, 50,25, 20, 15,10, 5, 2]
    xtks       = np.array([1/(t) for t in xpers])
    
    
    fig, ax = plt.subplots(1, 1, figsize=(14.5, 4.5))
    
    for vv in range(nexps):
        
        #ax      = axs[vv]
        
        # Get Variables
        svarsin = specexp[vv]
        P    = svarsin['specs']
        freq = svarsin['freqs']
        
        if use_redn:
            cflab = "Red Noise"
            CCs  = svarsin['CCs']
        else:
            cflab = "White Noise"
            CCs  = wnexp[vv] #* dtplot
            
        #P, freq, CCs, dof, r1 = svarsin
        
        # Convert units
        freq     = freq[0, :] * dtplot
        P        = P / dtplot
        Cbase    = CCs.mean(0)[:, 0]/dtplot
        Cupbound = CCs.mean(0)[:, 1]/dtplot
    
        # Plot Axis
        mu    = P.mean(0)
        sigma = P.std(0)
        
        # Plot Spectra
        if plot_log:
            ax.loglog(freq, mu, c=expcols[vv], lw=2.5,
                    label=expnames[vv], marker=vmarkers[vv])
            # ax.fill_between(freq, mu-sigma, mu+sigma,
            #                 color=expcols[vv], alpha=0.2, label="")
        else:
            ax.plot(freq, mu, c=expcols[vv], lw=2.5,
                    label=expnames[vv], marker=vmarkers[vv])
            # ax.fill_between(freq, mu-sigma, mu+sigma,
            #                 color=vcolors[vv], alpha=0.2, label="")
        
        if plot_ens:
            for e in range(nens):
                plotens = P[e,:]
                if e == 0 and vv == 0:
                    lbl = "Indv. Ens"
                else:
                    lbl = ""
                ax.plot(freq, plotens, c=vcolors[vv], lw=1,
                        label=lbl,alpha=0.08,zorder=-1)
        
        # Plot CCs
        if vv ==0:
            labc1 = cflab
            labc2 = "95% Confidence"
        else:
            labc1=""
            labc2=""
        if use_redn:
            ax.plot(freq, Cbase, color=expcols[vv], ls='solid', lw=1.2, label=labc1)
        ax.plot(freq, Cupbound, color=expcols[vv], ls="dotted",
                lw=2, label=labc2)
        
        
        #ax.axhline(freq,(freqs[-1]-freqs[0])*np.std(ts))
        
    
        # Set Labels and Ticks
        
        ax.legend()
        #ax.set_xticks(xtks, labels=xpers) # Uncomment ot Check
        ax.axvline([1/86],c="k",lw=.75)
        ax.axvline([1/2],c="k",lw=.75)
        ax.set_xlim([xtks[0], xtks[-1]])
        ax.set_xlabel("Cycles/Year")
        ax.set_ylabel("(%s)$^2$/cpy" % vunits[iv])
        
        # Twin upper axis
        ax2 = ax.twiny()
        if plot_log:
            ax2.set_xscale('log')
            
        ax2.set_xticks(xtks,labels=xpers)
        ax2.set_xlim([xtks[0], xtks[-1]])
        ax2.set_xlabel("Period (Years)")
        
        if vv == nvars-1:
            ax.set_xlabel("Period (Years)")
            
            
    #ax.set_ylim([0,3000])
    print(iv)
    print(vnames_long[iv])
    plt.suptitle("Power Spectra for %s\n nsmooth=%i, taper=%.2f" % (vnames_long[iv],nsmooth,pct),fontsize=20,y=1.1)
    
    savename = "%sAtmo_Spectra_%s_Annual_%s_%s_log%i.png" % (figpath, locfn,vname,specstr,plot_log)
    plt.savefig(savename, dpi=150, bbox_inches='tight', transparent=True)
    
    #% Do same plot as above, but ACF --------
    
    
    if calc_acf:
        xtks = np.arange(0, 37, 3)
        
        for kmonth in range(12):
            
            title = "%s ACF" % (mons3[kmonth])
            print(kmonth)
            fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(12, 4))
            ax, _   = viz.init_acplot(kmonth, xtks, lags, ax=ax, title=title)
            
            for vv in range(nexps):
                
                plotvar = acfexp[vv][kmonth,:,:]#dspt[vv].isel(mons=kmonth).squeeze()
                mu      = plotvar.mean(0)
                sigma   = plotvar.std(0)
                
                ax.plot(
                    lags, mu, c=expcols[vv], marker=vmarkers[vv], lw=2.5, label=expnames[vv])
                ax.fill_between(lags, mu-sigma, mu+sigma,
                                color=vcolors[vv], alpha=0.1, label="")
        
            ax.legend()
            ax.axhline([0], ls='solid', lw=0.75, color="k")
        
            savename = "%sAtmo_ACFs_%s_mon%02i.png" % (figpath, vname, kmonth+1)
            plt.savefig(savename, dpi=150, bbox_inches='tight', transparent=True)
        
    
#%% Spectra Debugging

nens    = 42

ts_test = dsatm_all[0].U
ts_test = preprocess_ds(ts_test)
tsann   = ts_test.groupby('time.year').mean('time')

tsin    = np.array([tsann.isel(ens=e).values for e in range(nens)]).flatten() #ts_in = ts_in

nsmooth = 100
pct     = 0
dtin    = 3600*24*365
specout = scm.quick_spectrum([tsin,],nsmooth=nsmooth,pct=pct,make_arr=True,
                             return_dict=True,dt=dtin)

#%% Copied function from viz_regional_spectra

def init_logspec(nrows,ncols,figsize=(10,4.5),ax=None,
                 xtks=None,dtplot=None,
                 fsz_axis=16,fsz_ticks=14,toplab=True,botlab=True):
    if dtplot is None:
        dtplot     = 3600*24*365  # Assume Annual data
    if xtks is None:
        xpers      = [100, 50,25, 20, 15,10, 5, 2]
        xtks       = np.array([1/(t) for t in xpers])
    
    if ax is None:
        newfig = True
        fig,ax = plt.subplots(nrows,ncols,constrained_layout=True,figsize=figsize)
    else:
        newfig = False
        
    ax = viz.add_ticks(ax)
    
    #ax.set_xticks(xtks,labels=xpers)
    ax.set_xscale('log')
    ax.set_xlim([xtks[0], xtks[-1]])
    if botlab:
        ax.set_xlabel("Frequency (Cycles/Year)",fontsize=fsz_axis)
    ax.tick_params(labelsize=fsz_ticks)
    
    ax2 = ax.twiny()
    ax2.set_xscale('log')
    ax2.set_xticks(xtks,labels=xpers,fontsize=fsz_ticks)
    ax2.set_xlim([xtks[0], xtks[-1]])
    if toplab:
        ax2.set_xlabel("Period (Years)",fontsize=fsz_axis)
    ax2.grid(True,ls='dotted',c="gray")
    
    if newfig:
        return fig,ax
    return ax


#%% Generate White Noise

mciter = 10000

def intgr_spec(P,freq):
    dw    = freq[1:] - freq[:-1]
    intgr = (P[1:] *dw).sum() + dw[0] * P[0]
    return intgr

def wnspectra_h0(ts,nsmooth,pct,dim=0,clvl=[.95,],mciter=10000,return_wn=False,dt=3600*24*365):
    
    sigma = np.nanstd(ts,dim)
    N     = ts.shape[dim]
    print(sigma)
    # Generate confidence intervals above a white noise null hypothesis use monte carlo
    # where [sigma] is the standard deviation of the timeseries of length [N].
    # Generates [mciter] number of white noise timeseries of the same length...
    # spectra = []
    # specout = []
    wnts    = []
    #sigmas  = []
    for mc in tqdm(range(mciter)):
        
        wn = np.random.normal(0,sigma,N)
        wnts.append(wn)
    
    print(nsmooth)
    specout   = scm.quick_spectrum(wnts,nsmooth,pct,dt=dt,make_arr=True,return_dict=True)
    specs     = specout['specs']
    
    specsum = []
    for ii in range(mciter):
        ss = intgr_spec(specs[ii,:],specout['freqs'][0,:])
        specsum.append(ss)
        
    clvl_spec = np.quantile(specs,clvl,axis=0)
    
    if return_wn:
        return clvl_spec,specs,specout['freqs'][0,:],specsum
    return clvl_spec


#%% Run it
nsmooth=nsmooth
wnclvl,wnspecs,wnfreq,wnsum=wnspectra_h0(tsin,nsmooth,pct,return_wn=True,dt=dtin)

#%%

import scipy as sp

# plotting Params
dtplot     = 3600*24*365
xpers      = [100, 50,25, 20, 15,10, 5, 2]
xtks       = np.array([1/(t) for t in xpers])
fsz_title  = 24 
fsz_ticks  = 16

# Get Plotting Spec/Freq
P      = specout['specs'].squeeze() / dtplot
freq   = specout['freqs'].squeeze() * dtplot

# Recompte some WN parameters
sigma2 = np.var(tsin)
print(sigma2)
df     = freq[-1] - freq[0]


# Recompute White Noise based on z*
N    = len(tsin)
z95  = sp.stats.norm.ppf(.95)
wn95 = (sigma2/df + z95 * sigma2/df /np.sqrt(N))

# Prepare Wn from MC
wnfreq_plot= wnfreq * dtplot
wn_mc      = wnclvl.squeeze() / dtplot #


fig,axs = plt.subplots(2,1,constrained_layout=True,figsize=(12,10))


# Log Plot
ax = axs[0]
ax.set_title("Log-Log Plot",fontsize=fsz_title)
ax = init_logspec(1,1,ax=ax,)
ax.loglog(freq,P,label="Base")

# Plot White Noise H0
for mc in tqdm(range(0,mciter,100)):
    #print(mc)
    ax.loglog(wnfreq*dtplot,wnspecs[mc,:]/dtplot,alpha=0.05,c="gray",zorder=-3)

# Plot Conf
ax.loglog(wnfreq_plot,wn_mc,label="95% (MC)",c="r",ls='dotted')
ax.axhline([sigma2/df],ls='dashed',c='k',label="White Noise h0",lw=1.2)
ax.axhline([wn95],ls='solid',c='violet',label="95% (z*)",lw=.75)


# Linear Plot
ax = axs[1]
ax.set_title("Linear Plot",fontsize=fsz_title)
ax.plot(freq,P,label="Base")
for mc in tqdm(range(0,mciter,100)):
    ax.plot(wnfreq*dtplot,wnspecs[mc,:]/dtplot,alpha=0.05,c="gray",zorder=-3)
#tsann.isel(ens=0).values

# Plot Conf
ax.axhline([sigma2/df],ls='dashed',c='k',label="White Noise h0",lw=1.2)
ax.axhline([wn95],ls='solid',c='violet',label="95% (z*)",lw=.75)
ax.plot(wnfreq_plot,wn_mc,label="95% (MC)",c="r",ls='dotted')

# Set Ticks
ax.set_xticks(xtks,labels=xpers,fontsize=fsz_ticks)
ax.set_xlim([xtks[0],xtks[-1]])
ax.legend()

savename = "%sAtmo_spectra_U_testh0_nsmooth%03i.png" % (figpath,nsmooth)
plt.savefig(savename, dpi=150, bbox_inches='tight', transparent=True)

#%% Lets Check The integrals


wnbase = sigma2/df

dw    = freq[1:] - freq[:-1]

# Get sum...
intgr = (P[1:] *dw).sum() + dw[0] * P[0]

# Get wn sum
wn50    = np.quantile(wnspecs,.50,axis=0) / dtplot
freqwn  = wnfreq * dtplot
dwwn    = freqwn[1:] - freqwn[:-1]
intgrwn = (wn50[1:] * dwwn).sum() + dwwn[0] * wn50[0]


print("Integrated spectra is: %f" % intgr)
print("Actual variance is   : %f" % sigma2) 
print("WN variance (50p) is : %f" % intgrwn) 


"""

Leve of 95: 1.777470530232491

sigma2: 1.5179910478829677

z* s/sqrt(n) = 

"""

#df * sigma2
