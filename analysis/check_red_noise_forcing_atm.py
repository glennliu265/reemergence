#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Check proportion of variance contained in the low frequencies....

Created on Mon Sep 23 14:46:14 2024

@author: gliu
"""



import xarray as xr
import numpy as np
import matplotlib as mpl

import matplotlib.pyplot as plt
import sys
import glob
import os

import tqdm
import time

import cartopy.crs as ccrs
import matplotlib.patheffects as pe
# ----------------------------------
# %% Import custom modules and paths
# ----------------------------------

# Import re-eergemce parameters

# Indicate the Machine!
machine = "Astraeus"

# First Load the Parameter File
cwd = os.getcwd()
sys.path.append(cwd+ "/..")
import reemergence_params as rparams

# Paths and Load Modules
pathdict = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])

# Set needed paths
figpath     = pathdict['figpath']
input_path  = pathdict['input_path']
output_path = pathdict['output_path']
procpath    = pathdict['procpath']
rawpath     = pathdict['raw_path']

#%% Import Custom Modules

# Import AMV Calculation
from amv import proc,viz
import amv.loaders as dl

# Import stochastic model scripts
import scm

#%% Plotting Information

bboxplot                    = [-80,0,20,65]
mpl.rcParams['font.family'] = 'Avenir'
mons3                       = proc.get_monstr(nletters=3)
fsz_tick                    = 18
fsz_axis                    = 22
fsz_title                   = 28

proj                        = ccrs.PlateCarree()

#%% Load necessary data

ds_uvel,ds_vvel = dl.load_current()
ds_bsf          = dl.load_bsf(ensavg=False)
ds_ssh          = dl.load_bsf(ensavg=False,ssh=True)

# Convert Currents to m/sec instead of cmsec
ds_uvel = ds_uvel/100
ds_vvel = ds_vvel/100

# Load data processed by [calc_monmean_CESM1.py]
ds_sss          = dl.load_monmean('SSS')
ds_sst          = dl.load_monmean('SST')


tlon  = ds_uvel.TLONG.mean('ens').values
tlat  = ds_uvel.TLAT.mean('ens').values


# Load Mixed-Layer Depth
mldpath = input_path + "mld/"
mldnc   = "CESM1_HTR_FULL_HMXL_NAtl_EnsAvg.nc"
ds_mld  = xr.open_dataset(mldpath+mldnc).h.load()

#ds_h          = dl.load_monmean('HMXL')

#%% Compute the velocity

ds_umod = (ds_uvel.UVEL ** 2 + ds_vvel.VVEL ** 2)**(0.5)


#%% Load Masks

# Load Land Ice Mask
icemask     = xr.open_dataset(input_path + "masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")


mask        = icemask.MASK.squeeze()
mask_plot   = xr.where(np.isnan(mask),0,mask)#mask.copy()


mask_reg_sub    = proc.sel_region_xr(mask,bboxplot)
mask_reg_ori    = xr.ones_like(mask) * 0
mask_reg        = mask_reg_ori + mask_reg_sub


mask_apply  = icemask.MASK.squeeze().values
#mask_plot[np.isnan(mask)] = 0

# Load Gulf Stream
ds_gs   = dl.load_gs()
ds_gs   = ds_gs.sel(lon=slice(-90,-50))
ds_gs2  = dl.load_gs(load_u2=True)

#%% Other Functions

def anomalize(ds):
    ds = ds - ds.mean('ens')
    ds = proc.xrdeseason(ds)
    return ds


#%% Lets check Fprime first (as this should impact both SST and SSS)


st          = time.time()
ncfp        = "CESM1_HTR_FULL_Fprime_timeseries_nomasklag1_nroll0_NAtl.nc"
dsfp        = xr.open_dataset(rawpath + ncfp).load()
print("Loaded data in %.2fs" % (time.time()-st))

dsfpanom    = anomalize(dsfp)

#%% Copy from apply_hpf


cutoff    = 120 # In Months
cutoffstr = "lpf%03imons" % (cutoff)
lopass    = lambda x: proc.lp_butter(x,cutoff,6,btype='lowpass')

st = time.time()
lpout = xr.apply_ufunc(
    lopass,
    dsfpanom,
    input_core_dims=[['time']],
    output_core_dims=[['time']],
    vectorize=True, 
    )
print("Applied Low Pass Filter in %.2fs" % (time.time()-st))
# Took 55.43 sec

#lpout = hpout.copy()


#%% Now compute the fraction of variance in low frequencies

varfp    = dsfpanom.Fprime.var('time')
varfp_lp = lpout.Fprime.var('time')

#%% Check a point

fig,ax  = plt.subplots(1,1)
tspt    = proc.selpt_ds(dsfpanom.Fprime,-30,50).isel(ens=0)
tspt_lp = proc.selpt_ds(lpout.Fprime,-30,50).isel(ens=0)

ax.plot(tspt,label='raw')
ax.plot(tspt_lp,label='lpf')
ax.legend()
ax.set_title("Var LP: %.2f, Var Raw: %.2f" % (np.var(tspt_lp),np.var(tspt)))

#%% plot the ratio

cints = np.arange(0,5.2,.2)
pmesh = False

fig,ax,_    = viz.init_orthomap(1,1,bboxplot,figsize=(12,4))
ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")

plotvar     = (varfp_lp / varfp).mean('ens') * 100

if pmesh:
    pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj)
    
else:
    
    pcm     = ax.contourf(plotvar.lon,plotvar.lat,plotvar,levels=cints,transform=proj)
    cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,
                         levels=cints,colors='w',
                         linewidths=0.75,transform=proj)
    
    ax.clabel(cl,fontsize=fsz_tick)
    
cb = viz.hcbar(pcm,ax=ax)#fig.colorbar(pcm,ax=ax)
cb.set_label("Fraction of Variance At Frequencies Below %i months" % (cutoff) + " (%)")

savename = "%sFprime_LPF_Ratio_cutoff%03imons.png" % (figpath,cutoff)
plt.savefig(savename,dpi=150,bbox_inches='tight')