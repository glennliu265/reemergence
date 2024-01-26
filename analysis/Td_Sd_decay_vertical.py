#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Examine Sd, Td decay at a point

Works with output grabbed from 

Created on Wed Jan 24 18:26:43 2024

@author: gliu

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

# Location
lonf = 330
latf = 50
locfn, loctitle = proc.make_locstring(lonf, latf)

datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/lon%s_lat%s/" % (
    lonf, latf)
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20240126/"
proc.makedir(figpath)


outpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/%s/" % locfn

# %% Load necessary files (see repair_file_SALT_CESM1.py)

ncsalt = outpath + "CESM1_htr_SALT_repaired.nc"

ds_salt = xr.open_dataset(ncsalt)

z = ds_salt.z_t.values  # /100 NOTE THIS WAS DONE IN repair code
times = ds_salt.time.values
salt = ds_salt.SALT.values  # [Ens x Time x Depth ]
nens, ntime, nz = salt.shape

timesstr = ["%04i-%02i" % (t.year, t.month) for t in times]

ens = np.arange(nens)+1

# %% Quick Hovmuller (it appears that these are full anomalies)
xtks = np.arange(0, ntime+1, 120)
e = 0
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
pcm = ax.pcolormesh(timesstr, z, salt[e, :, :].T, cmap="cmo.balance")
fig.colorbar(pcm)
ax.set_xticks(xtks)
ax.set_ylim(0, 3000)
ax.invert_yaxis()
ax.set_title("SALT in CESM1-PiControl (1920-2005)")


# %% Given the raw variables


# %% Compute the seasonal cycle

# Get Seasonal Cycle
scycle, tsmonyr = proc.calc_clim(
    salt, 1, returnts=True)  # [ens x yr x mon x z]

# Compute monthly anomaly
tsanom = tsmonyr - scycle[:, None, :, :]

# %% Remove the ensemble average

tsanom_ensavg = np.nanmean(tsanom, 0)
# plt.pcolormesh(timesstr,z,tsanom_ensavg.reshape(86*12,nz).T) #Visualize Ens Avg Pattern

tsanom_dt = tsanom - tsanom_ensavg[None, ...]
# Check detrending

iz = 0
e = 0
fig, ax = plt.subplots(1, 1)

ax.plot(tsanom[e, :, :, iz].flatten(), label="Raw", c='red')
ax.plot(tsanom_dt[e, :, :, iz].flatten(),
        label="Detrended", c='k', ls='dashed')
ax.plot(tsanom_ensavg[:, :, iz].flatten(), label="Ens. Avg.", c="mediumblue")
ax.legend()
ax.set_title("Detrended Value at Depth z=%im, Ens %i" % (z[iz], e+1))

# %% Remove NaNs

# z_nan = np.sum(tsanom_dt,(0,1,2))


# %% Compute Autocorrelation at each level


acfs_mon = []
lags = np.arange(0, 61, 1)
tsens = tsanom_dt[e, :, :, :]
for im in range(12):
    basemonth = im+1
    varin = tsanom_dt[e, ...].transpose(1, 0, 2)  # Month x Year x Npts
    out = proc.calc_lagcovar_nd(varin, varin, lags, basemonth, 1)
    acfs_mon.append(out)
acfs_mon = np.array(acfs_mon)  # [Mon Lag Depth]

# %% Visualize some things <o> <o> <o>

zz = 2

plotzz = [0, 10, 25, 35, 45, 49]
kmonth = 6
xtksl = np.arange(0, 66, 6)
fig, ax = viz.init_acplot(kmonth, xtksl, lags)
for zz in range(len(plotzz)):
    iz = plotzz[zz]
    ax.plot(lags, acfs_mon[kmonth, :, iz], label="z=%.2fm" %
            (z[iz]))  # alpha=.01 + .9*(zz/nz),color='gray')
ax.legend()

# %% Fit Exponential Function... (test plot)
zz = 2
kmonth = 2
lagmax = 3

acf_in = acfs_mon[kmonth, :, zz]

outdict = proc.expfit(acf_in, lags, lagmax=lagmax)

fig, ax = viz.init_acplot(kmonth, xtksl, lags)
ax.plot(lags, acf_in, label="ACF")
ax.plot(lags, outdict['acf_fit'], label=r"$\tau=%.2f$" %
        outdict['tau_inv'], ls='dashed')
ax.legend()


# %% Determine the Timescale for each month, depth level, and lagmax

nlags = len(lags)
lagmaxes = [1, 2, 3]
nlgm = len(lagmaxes)
tau_est = np.zeros((nlgm, 12, nz))
acf_est = np.zeros((nlgm, 12, nlags, nz))
for im in range(12):
    for zz in tqdm(range(nz)):
        acf_in = acfs_mon[im, :, zz]
        for lm in range(nlgm):

            lagmax = lagmaxes[lm]
            outdict = proc.expfit(acf_in, lags, lagmax=lagmax)
            tau_est[lm, im, zz] = outdict['tau_inv'].copy()
            acf_est[lm, im, :, zz] = outdict['acf_fit'].copy()

# %% Get HMXL/HBLT at the point
mldpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/CESM_proc/"
mldnc = "HBLT_FULL_HTR_lon-80to0_lat0to65_DTFalse.nc"
dsh = xr.open_dataset(mldpath+mldnc)
hbltpt = dsh.sel(lon=lonf-360, lat=latf,
                 method='nearest').load()  # [Ens time z_t]
hclim = hbltpt.groupby('time.month').mean(
    'time').squeeze().HBLT.values/100  # Ens x month, convert cm --> m

kprev, _ = scm.find_kprev(hclim.mean(1))
hmax = hclim.mean(1).max()

# %% Visualize kprev

mons3 = proc.get_monstr(nletters=3)
xlabs = mons3 + mons3[:1]

fig, ax = plt.subplots(1, 1, constrained_layout=True)

viz.viz_kprev(hclim.mean(1), kprev, locstring=loctitle, ax=ax,
              msize=10, mstyle="X", lw=2.5, txtalpha=.55, usetitle=False)
# ax.set_xlim([1,14])
# ax,set_
ax.invert_yaxis()
ax.set_xticklabels(xlabs)
ax.set_title("HBLT Cycle @ %s" % loctitle)
# ax.spines[[ 'top']].set_visible(False)
savename = "%sKprev_CESM_HTR_FULL.png" % figpath
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Retrieve damping for each month

debug = False
lm    = 0
taudt = scm.calc_tau_detrain(hclim.mean(1),kprev,z,tau_est[lm,...],debug=True)


#%% Visualize Detrainment
fig,ax =plt.subplots(1,1)
ax.plot(mons3,taudt,label="Monthy Sd', Function")
#ax.plot(mons3,tau_detrain_nonfunction,label="Monthly Sd, In Script",ls='dotted') # Confirmed that function was ok!
ax.legend()
# %%
