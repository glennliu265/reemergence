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
          "Umod",]

vnames_long = [
    "$SST$",
    "$Q_{net}$",
    "Stochastic Heat Flux",
    "Wind Modulus",
]

vunits = [
    "$\degree C$",
    "$W m^{-2}$",
    "$W m^{-2}$",
    "m $s^{-1}$",
]

vcolors = ["k",
           "royalblue",
           "violet",
           "limegreen"]

vmarkers = ["o",
            "x",
            "+",
            "d"]


ptpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/lon330_lat50/"

# %% Load the variables (basinwide)

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


# %% Plot variables at a point

lonf = -30
latf = 50
dspt = [proc.selpt_ds(ds, lonf, latf)
        for ds in ds_all]  # [Ens x Mon x Thres x Lag]
locfn, loctitle = proc.make_locstring(lonf, latf, lon360=True)

# %% Load data for the pint

ptpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/%s/" % locfn
ptnc = "CESM1_HTR_FULL_NAtl_AtmoVar_lon330_lat50.nc"
dsatm = xr.open_dataset(ptpath + ptnc).load()


# %%

# Plotting Parameters
ds = ds_all[0]
mpl.rcParams['font.family'] = 'JetBrains Mono'
bboxplot = [-80, 0, 20, 65]
proj = ccrs.PlateCarree()
lon = ds.lon.values
lat = ds.lat.values
mons3 = proc.get_monstr()


# %% Plot the ACFs for different atmospheric variables


xtks = np.arange(0, 61, 6)
lags = np.arange(0, 61, 1)

for kmonth in range(12):

    title = "%s ACF @ %s" % (mons3[kmonth], loctitle)

    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(12, 4))
    ax, _ = viz.init_acplot(kmonth, xtks, lags, ax=ax, title=title)

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

# %% Take the data and compute the spectra (usuing xarray ufuncs)

ts = dsatm['SST'].isel(ens=0).values


opt = 1
nsmooth = 10
dt = None
clvl = [.95,]


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


specatm = []
for vv in tqdm(range(nvars)):

    tsens = dsatm[vnames[vv]]
    tsens = preprocess_ds(tsens)

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

#%% Read out to numpy arrays and redo
nsmooth = 1
pct = 0.0
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
    
    