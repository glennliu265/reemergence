#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimate Exponential Point and compute 

Created on Wed Sep  6 09:10:59 2023

@author: gliu
"""

import xarray as xr
import numpy as np
import scipy as sp
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

#%%


datpath_ac = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
lonf       = -30
latf       = 50
varnames   = ("SST","SSS")
figpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20230914/"

#%% Import Custom Modules

sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
import scm
#%%

# Read in datasets
ds_pt = []
ac_pt = []
for v in range(2):
    ds = xr.open_dataset("%sHTR-FULL_%s_autocorrelation_thres0.nc" % (datpath_ac,varnames[v]))
    ds  = ds.sel(lon=lonf,lat=latf,method='nearest').sel(thres="ALL").load()# [ens lag mon]
    ds_pt.append(ds)
    ac_pt.append(ds[varnames[v]].values)

#%% Get some parameters

lags  = ds_pt[0].lag.values
nlags = len(lags)


#%% Fit an Exponential Curve

e  = 0
im = 1

# Select an ensemble member and month
acs_in = [ac[e,:,im] for ac in ac_pt]

#%% Try SciPy Curve Fitting
# https://stackoverflow.com/questions/3433486/how-to-do-exponential-and-logarithmic-curve-fitting-in-python-i-found-only-poly
expf  = lambda t,a,b,c: a*np.exp(-b*t)+c # A * exp(bt) + c 
expf2 = lambda t,a,b: a*np.exp(-b*t)     # No c
expf3 = lambda t,b: np.exp(b*t)         # No c and A

# Chose the func
funcin = expf3

# Choose the max lag to fit to
lagmaxes = [7,13,37]
lms      = len(lagmaxes)


# Fit each one
lbds_fit = []
for lm in range(lms):
    
    lagmax=lagmaxes[lm]
    
    # Fit Pointwise
    est_lbd = []
    for ii in range(2):
        
        x          = ds_pt[0].lag.values
        y          = acs_in[ii]
        popt, pcov = sp.optimize.curve_fit(funcin, x[:lagmax], y[:lagmax])
        
        # Compute Estimated Fit
        fity       = funcin(lags,*popt)
        est_lbd.append(popt)
        print(est_lbd)
        
        # Plot
        fig,ax = plt.subplots(1,1)
        ax.plot(lags,y,label="y")
        ax.plot(lags,fity,label="y-fit")
        ax.legend()
        ax.set_title("%s (b=%f)" % (varnames[ii],popt))
    lbds_fit.append(est_lbd)

#%% Visualize Exponential Fit for Each
lmcolors = ["limegreen","cornflowerblue","rebeccapurple"]
locfn,loctitle = proc.make_locstring(lonf,latf)
mons3 = proc.get_monstr(nletters=3)
xtks = np.arange(0,37,3)

v = 0

for v in range(2):
    title_plot = "%s ACF @ %s\n Basemonth: %s | Ens: %02i " % (varnames[v],loctitle,mons3[im],e+1)
    title_file = "%s_ACF_%s_mon%02i_ens%02i" % (varnames[v],locfn,im+1,e+1)
    
    fig,ax = plt.subplots(1,1,constrained_layout=True)
    ax.set_xticks(xtks)
    ax     = viz.add_ticks(ax)
    y      = acs_in[v]
    ax.plot(lags,y,label="Original ACF",color="k",marker="o")
    
    for lm in range(lms):
        lagmax = lagmaxes[lm]
        popt = lbds_fit[lm][v]
        fity       = funcin(lags,*popt)
        ax.plot(lags,fity,label="%i-Lag fit (%.1f months)" % (lagmax-1,1/popt[0]*-1),color=lmcolors[lm])
        ax.axvline(lags[lagmax-1],color=lmcolors[lm])
    ax.legend()
    ax.set_title(title_plot)
    ax.set_ylabel("Correlation")
    ax.set_xlabel("Lag (Months, Lag 0 = %s)" % mons3[im])
    
    savename = "%s%s.png" % (figpath,title_file)
    print(savename)
    plt.savefig(savename,bbox_inches="tight",dpi=150)



#%% Run a vanilla stochastic model

# Set Params
nt         = 120000
tnoise     = np.random.normal(0,1,nt)
basemonth  = 0
detrendopt = 1

# Other Parameters
dt         = 3600*24*30
h          = 50
cp         = 3850
rho        = 1026

# Set Equation
smint = lambda t,T0,lbd,F: T0*np.exp(lbd) + F

# Loop and integrate
ts_out = np.zeros(nt)
lbd_in = est_lbd[ii]#/(rho*cp*h) * dt
#lbd_in = -22 / (rho*cp*h) * dt
#lbd_in = -0.000018822
for t in tqdm(range(nt)):
    if t == 0:
        T0 = 0
    else:
        T0 = ts_out[t-1]
    F         = tnoise[t]#/(rho*cp*h)*dt
    ts_out[t] = smint(t,T0,lbd_in,F)

# Compute ACF
ts_out_myr = ts_out.reshape(int(nt/12),12).T
ac_sm      = proc.calc_lagcovar(ts_out_myr,ts_out_myr,lags,basemonth,detrendopt)

# # Primitive computation
# ac_sm = []
# for l in range(nlags):
#     ac_sm.append(np.corrcoef(ts_out[l:],ts_out[:(nt-l)])[0,1])

# Compare
# Plot
fig,ax = plt.subplots(1,1)
ax.plot(x,y,label="y")
ax.plot(x,fity,label="y-fit")
ax.plot(x,ac_sm,label="stochastic model")
ax.legend()
ax.set_title("%s (b=%f)" % (varnames[ii],popt))

#for l in range(nlags):
# Plot autocorrelation




