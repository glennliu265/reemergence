#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Quick script to check the Monhtly variance of Fprime and Qnet

Created on Mon Nov 18 15:57:52 2024

@author: gliu
"""


import numpy as np
import xarray as xr
import matplotlib.pyplot as plt



#%%

lonf = -65
latf = 36

rawpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"

nc_qnet = rawpath + "CESM1LE_qnet_NAtl_19200101_20050101_bilinear.nc"
nc_fprime = rawpath + "/monthly_variance/CESM1_HTR_FULL_Fprime_timeseries_nomasklag1_nroll0_NAtl_stdev.nc"


mpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/mld/"
hnc   = mpath + "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc"
ds_h  = xr.open_dataset(hnc).load()
hpt = proc.selpt_ds(ds_h.h,lonf,latf)


ds_fprime_std = xr.open_dataset(nc_fprime).load()
ds_qnet = xr.open_dataset(nc_qnet).load()



#%% Get the monhtly variance of nqnet


ds_qneta = proc.xrdeseason(ds_qnet.qnet)
ds_qneta = ds_qneta - ds_qneta.mean('ensemble')


#%%


qnet_pt = proc.selpt_ds(ds_qneta,lonf,latf).groupby('time.month').std('time')


fprime_pt = proc.selpt_ds(ds_fprime_std.Fprime,lonf,latf)
#%%

fig,ax = viz.init_monplot(1,1,)

ax.plot(mons3,qnet_pt.mean('ensemble'),label="Qnet")
ax.set_ylabel("Qnet")

ax.plot(mons3,fprime_pt.mean('ens')**.5,label="Fprime")
ax.set_ylabel("Fprime")
ax.legend()

#%% Get the Mixed-Layer Depth Information




fig,ax = viz.init_monplot(1,1,)

ax.plot(mons3,hpt,label="Qnet")
ax.set_ylabel("HMXL")


#%%


#%%

fig,ax = viz.init_monplot(1,1,)

ax.plot(mons3,qnet_pt.mean('ensemble').data / hpt.data,label="Qnet")
ax.set_ylabel("Qnet")

ax.plot(mons3,fprime_pt.mean('ens').data**.5 / hpt.data,label="Fprime")
ax.set_ylabel("Fprime")
ax.legend()
