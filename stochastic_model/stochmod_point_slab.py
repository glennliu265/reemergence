#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Try to get a correspondence between SLAB and Stochastic Model with Constant H

Copied upper section from stochmod_point

Created on Mon Dec 18 01:22:01 2023

@author: gliu

"""


import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
from tqdm import tqdm
import copy


from statsmodels.regression.linear_model import yule_walker as yl
from statsmodels.tsa.arima_process import arma_acf
#https://www.statsmodels.org/dev/generated/statsmodels.tsa.arima_process.arma_acf.html

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
lonf           = -30#-30
latf           = 50#50
locfn,loctitle = proc.make_locstring(lonf,latf)


rawpath        = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/"
datpath        = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/lon%s_lat%s/" % (lonf,latf)
figpath        = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20240108/"
proc.makedir(figpath)


#%% Get Parameters

# Load Lat/Lon and Find Point
lon,lat   = scm.load_latlon()
klon,klat = proc.find_latlon(lonf,latf,lon,lat)

# Load Fprime 
fname   = "Fprime_PIC_SLAB_rolln0.nc"
dsf     = xr.open_dataset(rawpath+"../"+fname)
Fprime  = dsf.sel(lon=lonf+360,lat=latf,method='nearest').Fprime.values # Time
ntime   = len(Fprime)
Fpt_new     = Fprime.reshape(int(ntime/12),12).std(0) 

# Load Qnet
qname     ="NHFLX_PIC_SLAB.nc"
dsq       = xr.open_dataset(rawpath+"../CESM_proc/"+qname)
qnetpt    = dsq.sel(lon=lonf+360,lat=latf,method='nearest').NHFLX.values # Time
qnetpt    = (qnetpt - qnetpt.mean(0)[None,:]).flatten() # Deseason

# Load Hblt
hname   = "SLAB_PIC_hblt.npy"
hblt    = np.load(rawpath+hname)
hblt_pt = hblt[klon,klat,:] 

# Load Damping
dampname = "SLAB_PIC_NHFLX_Damping_monwin3_sig005_dof893_mode5_lag1_ensorem0.npy"
damping  = np.load(rawpath+dampname) # Lon 180, Lat 180
hff_pt_new = damping[klon,klat,:]

# Load the SLAB SST
sstname  = "TS_anom_PIC_SLAB.nc"
dst      = xr.open_dataset(rawpath+"../CESM_proc/"+sstname)
sstpt    = dst.sel(lon=lonf+360,lat=latf,method='nearest').TS.values # Time

# Load Output from an Earlier Script (Note ONly for SPG Point 30W 50N)
outdir = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/debug_stochmod/"
savename = "%ssynth_stochmod_combine_output.npz" % outdir
npzsynth = np.load(savename)
synlabs = npzsynth['labels']
synmonvars  = npzsynth['monvars']
fpt_cv  = np.array([55.278503, 53.68089 , 42.456623, 33.448967, 22.954145, 22.506973,22.151728, 24.135042, 33.337887, 40.91648 , 44.905064, 51.132706])
damp_cv = np.array([18.79741573, 12.31693983,  9.71672964,  8.03764248,  7.99291682,6.53819919,  6.33767891,  8.54040241, 13.54183531, 19.00482941, 22.59606743, 22.18767834])

#%% Now.... Run the stochastic model? (non-entraining version)

dt      = 3600*24*30
T0      = 0
nyrs    = 10000
use_oldparam = False # Set to Use Old Parameters for damping/forcing experiments, SPG POINT ONLY

if use_oldparam:
    Fpt    = fpt_cv
    hff_pt = damp_cv
else:
    Fpt    = Fpt_new
    hff_pt = hff_pt_new
 
# Make Forcing
eta     = np.random.normal(0,1,nyrs*12)
forcing = eta * np.tile(Fpt,nyrs)

# Convert Stuff
lbd_a = scm.convert_Wm2(hff_pt,hblt_pt,dt)[None,None,:]
F     = scm.convert_Wm2(forcing,hblt_pt,dt)[None,None,:]
T,damping_term,forcing_term=scm.integrate_noentrain(lbd_a,F,T0=T0,multFAC=True,debug=True)

#%% Calculate AC, Do AR(1) Fit and get the damping

basemonth = 7
lagmax    = 3

lags      = np.arange(37)
xtks      = np.arange(0,39,3)

accesm    = scm.calc_autocorr([sstpt,],lags,basemonth)
expdict   = proc.expfit(accesm[0],lags,lagmax)

fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(6,4))

ax,_ = viz.init_acplot(basemonth-1,xtks,lags,)
ax.plot(lags,accesm[0],label="Slab",c='gray',lw=1.5,marker="o")
ax.plot(lags,expdict['acf_fit'],label=r"1-Lag Exp Fit ($\tau ^{-1}$ = %.2f mons)"% (-1*1/expdict['tau_inv']),ls='dashed')
ax.legend()

# Once confirmed, repeat for all basemonths
taufit = [proc.expfit(scm.calc_autocorr([sstpt,],lags,im+1)[0],lags,1)['tau_inv']*-1 for im in range(12)]

#%% Recompute Fprime

nyrs_slab  = int(Fprime.shape[0]/12)
taufit_conv = scm.convert_Wm2(np.array(taufit),hblt_pt,dt,reverse=True)
Fprime_fit = qnetpt + np.tile(taufit_conv,nyrs_slab) * np.roll(sstpt,1) # Roll
Ffitpt     = Fprime_fit.reshape(int(ntime/12),12).std(0) 


#%% Rerun Stochastic Model with R(1) Fit

forcing_new = eta * np.tile(Ffitpt,nyrs)
F_new       = scm.convert_Wm2(forcing_new,hblt_pt,dt)[None,None,:]
T_expfit,_,_= scm.integrate_noentrain(np.array(taufit)[None,None,:],F_new,T0=T0,multFAC=True,debug=True)

#%% Compare some basic metrics

ssts = [T.squeeze(),sstpt,T_expfit.squeeze()]
labels = ["SM","CESM-SLAB","SM (Exp Fit Damping)"]
colors = ["r","gray","brown"]

# ACF
acs       = scm.calc_autocorr(ssts,lags,basemonth)

# Monthly Variance
ntimes  = [int(sst.shape[0]/12) for sst in ssts]
monvars = [ssts[ii].reshape(ntimes[ii],12).var(0) for ii in range(len(ssts))] 

#%% Plot Monthly Variance

mons3  = proc.get_monstr(3)

fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(6,4))
for ii in range(3):
    
    ax.plot(mons3,monvars[ii],label=labels[ii],c=colors[ii],lw=1.5,marker="o")


ax.grid(True,ls='dotted')
ax.set_title("Monthly Variance")


# Add in old data
ax.plot(mons3,synmonvars[3],label=synlabs[3])
ax.plot(mons3,synmonvars[8],label=synlabs[8])


ax.legend()

#%% Plot ACF

fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(6,4))
ax,_ = viz.init_acplot(basemonth-1,xtks,lags)
for ii in range(3):
    
    ax.plot(lags,acs[ii],label=labels[ii],c=colors[ii],lw=1.5,marker="o")

ax.legend()
#%% Compare Parameters

fig,axs = plt.subplots(2,1,constrained_layout=True)

ax = axs[1]
ax.plot(mons3,Fpt,label="F' ($\lambda^a$)",color='r')
ax.plot(mons3,Ffitpt,label="F' ($Exp \,Fit$)",color='brown',ls='dashed')
ax.legend()
ax.set_ylabel("Forcing ($W m^{-2}$)")
ax.grid(True,ls='dashed')

ax = axs[0]
ax.plot(mons3,hff_pt,label="$\lambda^a$",color='r')
ax.plot(mons3,taufit_conv,label="Lag-1 Fit",color='brown',ls='dashed')
ax.set_ylabel("Heat Flux Feedback ($W m^{-2}$ K^{-1})")
ax.grid(True,ls='dashed')
ax.legend()


#%% Try Yule walker (Statsmodels)

# Test
im      = 5
sstcesm_mon = sstpt[im::12]

# Use Yule Walker to Estimate Coefficients
rhos,sigmas=yl(sstcesm_mon,order=1,method="mle")


theoacf = arma_acf(rhos,np.zeros(len(rhos)),lags=lags[-1])


#%% Try Yule walker (nitime)
import nitime


# Use Nitime to Estimate AR coefficients and compute the order

coefs,sigmas=nitime.algorithms.AR_est_YW(sstcesm_mon,12)
ntime       = 1000000
X_ar,noise,aph=nitime.utils.ar_generator(ntime,sigma=sigmas,coefs=coefs)
theoacf = nitime.utils.autocorr(X_ar)
plt.plot(theoacf),plt.xlim([0,37])


