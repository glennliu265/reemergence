#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Redo stochastic model anaylsis for all points within a single region.

Copied upper section of regional_analysis manual

Created on Thu Aug 29 06:57:07 2024

@author: gliu

"""


from amv import proc, viz
import scm
import amv.xrfunc as xrf
import amv.loaders as dl
import reemergence_params as rparams
import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import scipy as sp

import matplotlib.pyplot as plt
import sys
import glob
import os

import tqdm
import time

# ----------------------------------
# %% Import custom modules and paths
# ----------------------------------

# Import re-eergemce parameters

# Indicate the Machine!
machine = "Astraeus"

# First Load the Parameter File
cwd = os.getcwd()
sys.path.append(cwd + "/..")

# Paths and Load Modules
pathdict = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])

# Set needed paths
figpath = pathdict['figpath']
input_path = pathdict['input_path']
output_path = pathdict['output_path']
procpath = pathdict['procpath']
rawpath = pathdict['raw_path']


# %% Import Custom Modules

# Import AMV Calculation

# Import stochastic model scripts

proc.makedir(figpath)

# %%

bboxplot = [-80, 0, 20, 65]
mpl.rcParams['font.family'] = 'Avenir'
mons3 = proc.get_monstr(nletters=3)

fsz_tick = 18
fsz_axis = 20
fsz_title = 16

rhocrit = proc.ttest_rho(0.05, 2, 86)

proj = ccrs.PlateCarree()


# %%  Indicate Experients (copying upper setion of viz_regional_spectra )

# # #  Same as comparing lbd_e effect, but with Evaporation forcing corrections !!
regionset = "SSSCSU"
comparename = "SSS_Paper_Draft02"
expnames = ["SSS_Draft01_Rerun_QekCorr", "SSS_Draft01_Rerun_QekCorr_NoLbde",
            "SSS_Draft01_Rerun_QekCorr_NoLbde_NoLbdd", "SSS_CESM"]
expnames_long = ["Stochastic Model ($\lambda^e$, $\lambda^d$)",
                 "Stochastic Model ($\lambda^d$)", "Stochastic Model", "CESM1"]
expnames_short = ["SM_lbde", "SM_no_lbde", "SM_no_lbdd", "CESM"]
ecols = ["magenta", "forestgreen", "goldenrod", "k"]
els = ['dotted', "solid", 'dashed', 'solid']
emarkers = ['+', "d", "x", "o"]

# # # SST Comparison (Paper Draft, essentially Updated CSU) !!
# regionset       = "SSSCSU"
# comparename     = "SST_Paper_Draft02"
# expnames        = ["SST_Draft01_Rerun_QekCorr","SST_Draft01_Rerun_QekCorr_NoLbdd","SST_CESM"]
# expnames_long   = ["Stochastic Model","Stochastic Model (No $\lambda^d$)","CESM1"]
# expnames_short  = ["SM","SM_NoLbdd","CESM"]
# ecols           = ["forestgreen","goldenrod","k"]
# els             = ["solid",'dashed','solid']
# emarkers        = ["d","x","o"]


cesm_exps = ["SST_CESM", "SSS_CESM", "SST_cesm2_pic", "SST_cesm1_pic",
             "SST_cesm1le_5degbilinear", "SSS_cesm1le_5degbilinear",]
# %% Load the Dataset (us sm output loader)
# Hopefully this doesn't clog up the memory too much

nexps = len(expnames)
ds_all = []
for e in tqdm.tqdm(range(nexps)):

    # Get Experiment information
    expname = expnames[e]

    if "SSS" in expname:
        varname = "SSS"
    elif "SST" in expname:
        varname = "SST"

    # For stochastic model output
    ds = dl.load_smoutput(expname, output_path)

    if expname in cesm_exps:
        print("Detrending and deseasoning")
        ds = proc.xrdeseason(ds[varname])
        if 'ens' in list(ds.dims):
            ds = ds - ds.mean('ens')
        else:
            ds = proc.xrdetrend(ds)
        ds = xr.where(np.isnan(ds), 0, ds)  # Sub with zeros for now
    else:
        ds = ds[varname]

    ds_all.append(ds)
    
#%% Load the ACFs instead

acfs = []
for e in tqdm.tqdm(range(nexps)):
    
    # Get Experiment information
    expname = expnames[e]

    if "SSS" in expname:
        varname = "SSS"
    elif "SST" in expname:
        varname = "SST"

    ## For stochastic model output
    #ds = dl.load_smoutput(expname, output_path)

    if expname in cesm_exps:
        ncname = "%sCESM1_1920to2005_%sACF_lag00to60_ALL_ensALL.nc" % (procpath,varname)
    else:
        ncname = "%sSM_%s_%s_autocorrelation_thresALL_lag00to60.nc" % (procpath,expname,varname,)
        
    ds = xr.open_dataset(ncname).load()
    acfs.append(ds)
    
# %% Load some variables to plot

# Load Current
ds_uvel, ds_vvel = dl.load_current()
uvel_regrid,vvel_regrid = dl.load_current(regrid=True)

# load SSS Re-emergence index (for background plot)
ds_rei = dl.load_rei("SSS_CESM", output_path).load().rei

# Load Gulf Stream
ds_gs = dl.load_gs()
ds_gs = ds_gs.sel(lon=slice(-90, -50))

# Load 5deg mask
maskpath = input_path + "masks/"
masknc5 = "cesm1_htr_5degbilinear_icemask_05p_year1920to2005_enssum.nc"
dsmask5 = xr.open_dataset(maskpath + masknc5)
dsmask5 = proc.lon360to180_xr(dsmask5).mask.drop_duplicates('lon')

masknc = "CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc"
dsmask = xr.open_dataset(maskpath + masknc).MASK.load()

maskin = dsmask
ds_gs2 = dl.load_gs(load_u2=True)

# Load Land Ice Mask
icemask     = xr.open_dataset(input_path + "masks/CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc")


mask        = icemask.MASK.squeeze()
mask_plot   = xr.where(np.isnan(mask),0,mask)#mask.copy()


mask_reg_sub    = proc.sel_region_xr(mask,bboxplot)
mask_reg_ori    = xr.ones_like(mask) * 0
mask_reg        = mask_reg_ori + mask_reg_sub


mask_apply  = icemask.MASK.squeeze().values

# %% Plot Locator and Bounding Box w.r.t. the currents

sel_box   = [-70,-55,35,40] # Sargasso Sea SSS CSU
bbname = "Sargasso Sea"
lonf = -65
latf = 36


bbname     = "North Atlantic Current"
sel_box    =  [-40,-30,40,50] # NAC
lonf       = -34
latf       = 46

# sel_box = [-40, -25, 50, 60]  # Irminger
# bbname = "Irminger Sea"
# lonf = -36
# latf = 58


# sel_box    = [-37,-25,50,60] # yeager 2012 SPG

bbfn, bbti = proc.make_locstring_bbox(sel_box)

# sel_box = [-40,-25,50,60]
# sel_box   = [-45,-38,20,25] # Azores High Proximity


qint = 2

# Restrict REI for plotting
selmons = [1, 2]
iyr = 0
plot_rei = ds_rei.isel(mon=selmons, yr=iyr).mean('mon').mean('ens')
rei_cints = np.arange(0, 0.55, 0.05)
rei_cmap = 'cmo.deep'

# Initialize Plot and Map
fig, ax, _ = viz.init_orthomap(1, 1, bboxplot, figsize=(18, 6.5))
ax = viz.add_coast_grid(ax, bbox=bboxplot, fill_color="lightgray")


# Plot Currents
plotu = ds_uvel.UVEL.mean('ens').mean('month').values
plotv = ds_vvel.VVEL.mean('ens').mean('month').values
tlon = ds_uvel.TLONG.mean('ens').data
tlat = ds_uvel.TLAT.mean('ens').data

ax.quiver(tlon[::qint, ::qint], tlat[::qint, ::qint], plotu[::qint, ::qint], plotv[::qint, ::qint],
          color='navy', transform=proj, alpha=0.75)

l1 = viz.plot_box(sel_box)

# Plot Re-emergence INdex
ax.contourf(plot_rei.lon, plot_rei.lat, plot_rei,
            cmap='cmo.deep', transform=proj, zorder=-1)


# Plot Gulf Stream Position
ax.plot(ds_gs.lon, ds_gs.lat.mean('ens'), transform=proj, lw=1.75, c="k")


ax.set_title("Bounding Box Test: %s" % (str(sel_box)), fontsize=fsz_title)


# %% Perform Regional Subsetting and Analysis
ds_all = [proc.format_ds_dims(ds) for ds in ds_all]
acf_all = [proc.format_ds_dims(ds) for ds in acfs]

dsreg = [proc.sel_region_xr(ds, sel_box) for ds in ds_all]
regavg_ts = [ds.mean('lat').mean('lon').data for ds in dsreg]


acfreg = [proc.sel_region_xr(ds,sel_box) for ds in acf_all]

# Repair ACFs to have variable named "acf" instead of varname
def repair_acfname(ds,varname):
    vnames = list(ds.keys())
    dimnames = list(ds.dims)
    print(vnames)
    print(dimnames)
    if varname in vnames:
        print("Renaming %s" % varname)
        ds = ds.rename({varname:"acf"})
    if "acfs" in vnames:
        ds = ds.rename({"acfs":"acf"})
        print("Renaming <acfs>")
    if 'mons' in dimnames:
        ds = ds.rename({'mons':'basemon'})
        print("Renaming <mons>")
    if 'month' in dimnames:
        ds = ds.rename({'month':'basemon'})
        print("Renaming <month>")
    ds = ds.sel(lags=slice(0,36))
    return ds
    
acfreg = [repair_acfname(ds,varname).acf.squeeze() for ds in acfreg]




# %% Write xr ufunc, apply over Lat/Lon/Run. Moved this to xrfunc

recalculate=False
# # Compute Monthly ACFs
# def compute_monthly_acf(tsin,nlags):

#     # Pointwise script acting on array [time]
#     ts = tsin.copy()

#     # Stupid fix, set NaNs to zeros
#     if np.any(np.isnan(ts)):
#         if np.all(np.isnan(tsin)):
#             return np.zeros((12,nlags)) * np.nan # Return all NaNs
#         nnan = np.isnan(tsin).sum()
#         print("Warning, NaN points found within timeseries. Setting NaN to zero" % nnan)
#         ts[np.isnan(ts)] = 0.

#     # Set up lags, separate to month x yrear
#     lags    = np.arange(nlags)
#     ntime   = len(ts)
#     nyr     = int(ntime/12)
#     tsmonyr = ts.reshape(nyr,12).T # Transpose to [month x year]

#     # Preallocate
#     sst_acfs = np.zeros((12,nlags)) * np.nan # base month x lag
#     for im in range(12):
#         ac= proc.calc_lagcovar(tsmonyr,tsmonyr,lags,im+1,0,yr_mask=None,debug=False)
#         sst_acfs[im,:] = ac.copy()
#     return sst_acfs


# st = time.time()
# #ds1  = dsreg[0] # Need to assign dummy ds here
# acffunc = lambda x: compute_monthly_acf(x,37)
# acfs = xr.apply_ufunc(
#     acffunc,
#     ds1,
#     input_core_dims=[['time']],
#     output_core_dims=[['basemon','lags']],
#     vectorize=True,
#     )
# print("Computed Pointwise ACF computation in %.2fs" % (time.time()-st))

nlags = 37
if recalculate:
    acfs_byexp = []
    for rr in tqdm.tqdm(range(nexps)):
        ds = dsreg[rr] * dsmask.squeeze()
    
        st = time.time()
        acfs = xrf.pointwise_acf(ds, nlags)
        acfs_byexp.append(acfs)
        print("Completed computations for experiment in %.2fs" % (time.time()-st))
    nens, nlat, nlon, _, _ = acfs_byexp[-1].shape
else:
    acfs_byexp = acfreg.copy()
    nlon,nlat,_,_=acfs_byexp[0].shape

# %% Plot the ensemble average mean ACF for each month


loopmon = [1,]

for kmonth in range(12):
    if kmonth not in loopmon:
        continue
    
    
    locfn,loctitle=proc.make_locstring(lonf,latf)
    
    lags = np.arange(nlags)
    xtks = lags[::2]
    
    fig, ax = plt.subplots(1, 1,constrained_layout=True,figsize=(10,6))
    ax, _ = viz.init_acplot(kmonth, xtks, lags, title=None)
    
    for ex in range(nexps):
        
        inacf = acfs_byexp[ex].isel(basemon=kmonth)#.mean('ens')
        if 'ens' in list(inacf.dims):
            inacf = inacf.mean('ens')
            
        # Plot the individual points
        for a in range(nlat):
            for o in range(nlon):
                plotacf= inacf.isel(lat=a,lon=o)
                ax.plot(lags,plotacf,c=ecols[ex],alpha=0.1)
        
        # Plot Region Mean
        plotacf = inacf.mean('lat').mean('lon')
        ax.plot(lags,plotacf,c=ecols[ex],alpha=1,label=expnames_long[ex] + ", Region Mean ACF",lw=2.5)
        
        # Plot the selected point
        plotacf = inacf.sel(lon=lonf,lat=latf,method='nearest')
        ax.plot(lags,plotacf,c=ecols[ex],
                ls='dashed',marker="x",
                alpha=1,label=expnames_long[ex] + ", %s" % loctitle,lw=2.5)
        
        
    ax.legend(fontsize=8)
    ax.set_title("%s (%s)" % (bbname,bbti))
    ax.set_ylabel("Correlation with %s Anomalies" % (mons3[kmonth]))
    savename = "%sPointwise_ACF_%s_region_%s_%02i.png" % (figpath,comparename,bbname[:3],kmonth+1)
    plt.savefig(savename,dpi=150)
        
        
#%% Compute the monthly variance

monvar_byexp = [ds.groupby('time.month').var('time') for ds in dsreg]

#%% Plot the result



nens, _,nlat, nlon = monvar_byexp[0].shape

#vlm = [0,1]
#vlm = [0,0.5]
vlm = [0,0.05]

locfn,loctitle=proc.make_locstring(lonf,latf)

lags = np.arange(nlags)
xtks = lags[::2]

fig,ax = viz.init_monplot(1,1,figsize=(10,6))
#fig, ax = plt.subplots(1, 1,constrained_layout=True,figsize=(10,6))
#ax, _ = viz.init_acplot(kmonth, xtks, lags, title=None)

for ex in range(nexps):
    
    inacf = monvar_byexp[ex].mean('ens')
    
    # Plot the individual points
    for a in range(nlat):
        for o in range(nlon):
            plotacf= inacf.isel(lat=a,lon=o)
            ax.plot(mons3,plotacf,c=ecols[ex],alpha=0.1)
    
    # Plot Region Mean
    plotacf = inacf.mean('lat').mean('lon')
    ax.plot(mons3,plotacf,c=ecols[ex],alpha=1,label=expnames_long[ex] + ", Region Mean ACF",lw=2.5)
    
    # Plot the selected point
    plotacf = inacf.sel(lon=lonf,lat=latf,method='nearest')
    ax.plot(mons3,plotacf,c=ecols[ex],
            ls='dashed',marker="x",
            alpha=1,label=expnames_long[ex] + ", %s" % loctitle,lw=2.5)
    
ax.set_ylim(vlm)
ax.legend(fontsize=8)
ax.set_title("%s (%s)" % (bbname,bbti))
ax.set_ylabel("Correlation with %s Anomalies" % (mons3[kmonth]))
savename = "%sPointwise_Monvar_%s_region_%s.png" % (figpath,comparename,bbname[:3])
plt.savefig(savename,dpi=150)



#%% Try to filter points by mean currents


umod = (uvel_regrid.UVEL **2 + vvel_regrid.VVEL **2)**0.5
umod,_ = proc.resize_ds([umod,ds_all[0]])
umod_in = umod.mean('ens').mean('month') * mask

#%%
thres  = [0.25,0.5,0.75,0.90,0.95]
var_in = umod_in
use_quantile=True
# Function: Split Lon/Lat Map based on percentiles

var_flat    = var_in.data.flatten()
if use_quantile:
    
    thresvals   = np.nanquantile(var_flat,thres)
else:
    thresvals   = thres

# Make a map of the values
classmap = xr.zeros_like(var_in)


boolmap = []
nthres  = len(thres) + 1
for nn in range(nthres):
    print(nn)
    if nn == 0:
        print("x < %f" % (thresvals[nn]))
        #boolmap.append(var_in < thresvals[nn])
        classmap = xr.where(var_in < thresvals[nn],nn,classmap)
    elif nn == (nthres-1):
        print("x >= %f" % (thresvals[nn-1]))
        #boolmap.append(var_in >= thresvals[nn-1])
        classmap = xr.where(var_in >= thresvals[nn-1],nn,classmap)
    else:
        print("%f < x <= %f" % (thresvals[nn-1],thresvals[nn]))
        #boolmap.append( (var_in > thresvals[nn-1]) & (var_in <= thresvals[nn]))
        classmap = xr.where( (var_in > thresvals[nn-1]) & (var_in <= thresvals[nn]),nn,classmap)
    classmap.plot(),plt.title("nn=%i" % nn),plt.show()
    
# Start by filling NaN points
#

classmap.plot(),plt.show()

#umod_all = umod_in.data.flatten()
#%% Make the Above into a Function

def classify_bythres(var_in,thres,use_quantile=True,debug=False):
    
    if use_quantile: # Get Quantiles based on flattened Data
        var_flat    = var_in.data.flatten()
        thresvals   = np.nanquantile(var_flat,thres)
    else: # Or just used provided values
        thresvals   = thres

    # Make a map of the values
    classmap = xr.zeros_like(var_in)
    
    boolmap = []
    nthres  = len(thres) + 1
    for nn in range(nthres):
        print(nn)
        if nn == 0:
            print("x < %f" % (thresvals[nn]))
            #boolmap.append(var_in < thresvals[nn])
            classmap = xr.where(var_in < thresvals[nn],nn,classmap)
        elif nn == (nthres-1):
            print("x >= %f" % (thresvals[nn-1]))
            #boolmap.append(var_in >= thresvals[nn-1])
            classmap = xr.where(var_in >= thresvals[nn-1],nn,classmap)
        else:
            print("%f < x <= %f" % (thresvals[nn-1],thresvals[nn]))
            #boolmap.append( (var_in > thresvals[nn-1]) & (var_in <= thresvals[nn]))
            classmap = xr.where( (var_in > thresvals[nn-1]) & (var_in <= thresvals[nn]),nn,classmap)
        if debug:
            classmap.plot(),plt.title("nn=%i" % nn),plt.show()
    # Set NaN points to NaN
    classmap = xr.where(np.isnan(var_in),np.nan,classmap)
    if debug:
        classmap.plot(),plt.show()
    if use_quantile:
        return classmap,thresvals
    return classmap
        
thres  = [0.25,0.5,0.75,0.90,0.95]
var_in = umod_in
use_quantile=True
classmap,thresvals = classify_bythres(var_in,thres,use_quantile=use_quantile)
        




#%%



