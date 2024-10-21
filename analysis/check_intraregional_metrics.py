#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Redo stochastic model anaylsis for all points within a single region.

Copied upper section of regional_analysis manual

Created on Thu Aug 29 06:57:07 2024

@author: gliu

"""



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
import reemergence_params as rparams

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
from amv import proc, viz
import scm
import amv.xrfunc as xrf
import amv.loaders as dl

# Import stochastic model scripts

proc.makedir(figpath)

# %%

bboxplot        = [-80, 0, 20, 65]
mpl.rcParams['font.family'] = 'Avenir'
mons3           = proc.get_monstr(nletters=3)

fsz_tick        = 18
fsz_axis        = 20
fsz_title       = 16

rhocrit         = proc.ttest_rho(0.05, 2, 86)

proj            = ccrs.PlateCarree()


# %%  Indicate Experients (copying upper setion of viz_regional_spectra )

# # # #  Same as comparing lbd_e effect, but with Evaporation forcing corrections !!
# regionset = "SSSCSU"
# comparename = "SSS_Paper_Draft02"
# expnames = ["SSS_Draft01_Rerun_QekCorr", "SSS_Draft01_Rerun_QekCorr_NoLbde",
#             "SSS_Draft01_Rerun_QekCorr_NoLbde_NoLbdd", "SSS_CESM"]
# expnames_long = ["Stochastic Model ($\lambda^e$, $\lambda^d$)",
#                  "Stochastic Model ($\lambda^d$)", "Stochastic Model", "CESM1"]
# expnames_short = ["SM_lbde", "SM_no_lbde", "SM_no_lbdd", "CESM"]
# ecols = ["magenta", "forestgreen", "goldenrod", "k"]
# els = ['dotted', "solid", 'dashed', 'solid']
# emarkers = ['+', "d", "x", "o"]

# # SST Comparison (Paper Draft, essentially Updated CSU) !!
regionset       = "SSSCSU"
comparename     = "SST_Paper_Draft02"
expnames        = ["SST_Draft01_Rerun_QekCorr","SST_Draft01_Rerun_QekCorr_NoLbdd","SST_CESM"]
expnames_long   = ["Stochastic Model","Stochastic Model (No $\lambda^d$)","CESM1"]
expnames_short  = ["SM","SM_NoLbdd","CESM"]
ecols           = ["forestgreen","goldenrod","k"]
els             = ["solid",'dashed','solid']
emarkers        = ["d","x","o"]


cesm_exps = ["SST_CESM", "SSS_CESM", "SST_cesm2_pic", "SST_cesm1_pic",
             "SST_cesm1le_5degbilinear", "SSS_cesm1le_5degbilinear",]


# Load Regions
# Get Bounding Boxes
regionset       = "SSSCSU"
regiondicts     = rparams.region_sets[regionset]
bboxes          = regiondicts['bboxes']
regions_long    = regiondicts['regions_long']
rcols           = regiondicts['rcols']
rsty            = regiondicts['rsty']


# %% Load the Dataset (us sm output loader)
# Hopefully this doesn't clog up the memory too much

nexps   = len(expnames)
ds_all  = []
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

# # load SSS Re-emergence index (for background plot)
ds_rei = dl.load_rei("SSS_CESM", output_path).load().rei
# ds_rei_sss = dl.load_rei("SSS_CESM", output_path).load().rei
# ds_rei_sst = dl.load_rei("SST_CESM", output_path).load().rei

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

#%% Load REI From a specific run

rei_nc   = "REI_Pointwise.nc"
rei_path = output_path + "SSS_CESM/Metrics/"
ds_rei   = xr.open_dataset(rei_path + rei_nc).load().rei
reiplot  = ds_rei.isel(mon=[1,2],yr=0).mean('mon').mean('ens')

reiplot_sss= reiplot.copy()


rei_nc   = "REI_Pointwise.nc"
rei_path = output_path + "SST_CESM/Metrics/"
ds_rei_sst   = xr.open_dataset(rei_path + rei_nc).load().rei
reiplot_sst  = ds_rei_sst.isel(mon=[1,2],yr=0).mean('mon').mean('ens')


# %% Plot Locator and Bounding Box w.r.t. the currents

# sel_box   = [-70,-55,35,40] # Sargasso Sea SSS CSU
# bbname    = "Sargasso Sea"
# lonf      = -65
# latf      = 36

# bbname     = "North Atlantic Current"
# sel_box    =  [-40,-30,40,50] # NAC
# lonf       = -34
# latf       = 46

sel_box = [-40, -25, 50, 60]  # Irminger
bbname  = "Irminger Sea"
lonf    = -36
latf    = 58


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

# Subset SST/SSS and ACFs
ds_all  = [proc.format_ds_dims(ds) for ds in ds_all]
acf_all = [proc.format_ds_dims(ds) for ds in acfs]

# Take Regional Averages
dsreg       = [proc.sel_region_xr(ds, sel_box) for ds in ds_all]
regavg_ts   = [ds.mean('lat').mean('lon').data for ds in dsreg]
acfreg      = [proc.sel_region_xr(ds,sel_box) for ds in acf_all]

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
plt.savefig(savename,dpi=150,bbox_inches='tight')

# =============================================================================
#%% Part 2: Identifying Regimes, 
#%% Try to filter points by mean currents
# =============================================================================

# User Selection
use_quantile = True
thres        = [.25,.5,.75]

# Compute Modulus
umod    = (uvel_regrid.UVEL **2 + vvel_regrid.VVEL **2)**0.5
umod,_  = proc.resize_ds([umod,ds_all[0]])
umod_in = umod.mean('ens').mean('month') * mask

# Set input variable
classifier = umod_in.copy()
classifier_name = "umod"

# Restrict Umod to the region, then apply classification
classifier_reg,_          = proc.resize_ds([classifier,dsreg[0]])
classmap,thresval         = proc.classify_bythres(classifier_reg,thres,use_quantile=True,debug=True)

# Make the labels
threslabs = proc.make_thres_labels(thresval)
classes   = np.unique(classmap)
classes   = [c for c in classes if ~np.isnan(c)]
classcol  = ["darkred","orange","cornflowerblue",'navy']
classmarker = ["v","x","o","^"]
nclasses = len(classes)

#%% Visualize the Region/lcoator


cints_rei       = np.arange(0,1.02,0.02)

plot_SST_rei    = False

# bbnameshort     = "NAC"
# lonf = -39
# latf = 44



bbnameshort         = "IRM"

if bbnameshort == "IRM":
    # First POint (odd monvar)
    lonf                = -35
    latf                = 53
    # Second Point ()
    lonf = -36
    latf = 56
elif bbnameshort == "NAC":
    lonf = -39
    latf = 44
elif bbnameshort == "SAR":
    lonf      = -65
    latf      = 36
    

    
# bbnameshort = "SAR"


# Plot Target Metric

# Classification Output (Current Speed)
plotvar         = classifier_reg#umod_reg
plotname        = "Mean Surface Current Speed [cm/s]"
plotname_fn     = "Classification_Output"
#plotvar = monvar_byexp[ex].mean('ens').isel(month=6)#.max('month') - monvar_byexp[ex].mean('ens').min('month')

# # Monthly Variance in a month
# im              = 1
# plotvar         = monvar_byexp[ex].isel(month=im).mean('ens')
# plotname        = "Interannual Variability of %s Anomalies [%s]$^2$" % (mons3[im],vunit)
# plotname_fn     = "Monvar%02i" % (im+1)
# #plotvar = monvar_byexp[ex].mean('ens').isel(month=6)#.max('month') - monvar_byexp[ex].mean('ens').min('month')


# Make the Figure
fig,ax,bbplotr = viz.init_regplot(bbnameshort,)

# Plot Markers of each class
for o in range(nlon):
    for a in range(nlat):
        classpt = classmap.isel(lon=o,lat=a)
        if np.isnan(classpt):
            continue
        cid     = int(classpt.item())
        ax.plot(classpt.lon,classpt.lat,transform=proj,
                marker=classmarker[cid],c=classcol[cid],markersize=15,zorder=2)
        
# Plot Mean Currents
# Plot Currents
plotu = ds_uvel.UVEL.mean('ens').mean('month').values
plotv = ds_vvel.VVEL.mean('ens').mean('month').values
tlon = ds_uvel.TLONG.mean('ens').data
tlat = ds_uvel.TLAT.mean('ens').data
ax.quiver(tlon[::qint, ::qint], tlat[::qint, ::qint], plotu[::qint, ::qint], plotv[::qint, ::qint],
          color='navy', transform=proj, alpha=0.25)

# Plot the Target Variable
pcm         = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,zorder=-1)

# Colorbar Business
cb          = viz.hcbar(pcm,ax=ax,fraction=0.045)
cb.ax.tick_params(labelsize=fsz_tick)
cb.set_label(plotname,fontsize=fsz_axis)

# PLot REI
if varname == "SSS" and plot_SST_rei is False:
    plotvar = reiplot_sss * mask_reg
elif varname == "SST" or plot_SST_rei:
    plotvar = reiplot_sst * mask_reg
    
cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,
                levels=cints_rei,colors="k",lw=2.5,
                transform=proj)

ax.clabel(cl,fontsize=fsz_tick)

# Plot Labels for points
plotvar = proc.sel_region_xr(plotvar,bbplotr)
for (i, j), z in np.ndenumerate(plotvar):
    ax.text(plotvar.lon.data[j], plotvar.lat.data[i], '{:0.3f}'.format(z),
            ha='center', va='center',transform=proj,fontsize=9,color='lightgray',zorder=4,)#path_effects=[pe.withStroke(linewidth=1.5, foreground="w")])

# Select A Point
reipt = plotvar.sel(lon=lonf,lat=latf,method='nearest')
print(reipt)
ax.plot(reipt.lon,reipt.lat,marker="+",markersize=42,c="k",transform=proj)

savename = "%sLocator_%s_%s_region_%s.png" % (figpath,plotname_fn,comparename,bbname[:3])
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%%  Visualize target metrics as a scatterplot to see if there is a linear relationship

im          =  1

if "SSS" in expnames[ex]:
    vunit = "psu"
else:
    vunit = "degC"
    
for im in range(12):
    xvar        = umod_reg.data.flatten()
    xvarname    = "Current Speed (cm/sec)"
    yvar        = monvar_byexp[ex].isel(month=im).mean('ens').data.flatten()
    yvarname    = "Monthly Variance in %s [%s]$^2$" % (mons3[im],vunit)
    
    fig,ax = plt.subplots(1,1,constrained_layout=True)
    for cid in range(nclasses):
        
        
        plotvar = classmap.data.flatten()
        idclass = plotvar == cid
        
        ax.scatter(xvar[idclass],yvar[idclass],color=classcol[cid],marker=classmarker[cid])
    ax.set_xlabel(xvarname)
    ax.set_ylabel(yvarname)
    

#%% Now replot the ACFs, but with the class separation

kmonth = 1
ex     = 3



locfn,loctitle=proc.make_locstring(lonf,latf)

lags = np.arange(nlags)
xtks = lags[::2]

fig, ax = plt.subplots(1, 1,constrained_layout=True,figsize=(10,6))
ax, _ = viz.init_acplot(kmonth, xtks, lags, title=None)



inacf = acfs_byexp[ex].isel(basemon=kmonth)#.mean('ens')
if 'ens' in list(inacf.dims):
    inacf = inacf.mean('ens')

# Plot the individual points
class_avg =[[],[],[],[]]
for a in range(nlat):
    for o in range(nlon):

        classid = classmap.isel(lon=o,lat=a).data.item()
        
        if np.isnan(classid):
            print("Nan Detected")
            continue
        classid = int(classid)

        cmk   = classmarker[classid]
        plotc = classcol[classid]
        
        plotacf= inacf.isel(lat=a,lon=o)
        
        ax.plot(lags,plotacf,c=plotc,alpha=0.2,marker=cmk,label=None)
        
        class_avg[classid].append(plotacf)

# Plot Class Averages
for c in range(nclasses):
    print(c)
    plotacf = np.nanmean(np.array(class_avg[c]),0)
    ax.plot(lags,plotacf,c=classcol[c],alpha=1,marker=classmarker[c],label="%s" % threslabs[c])

# Plot Region Mean
plotacf = inacf.mean('lat').mean('lon')
ax.plot(lags,plotacf,c=ecols[ex],alpha=1,label=expnames_long[ex] + ", Region Mean ACF",lw=2.5)

# Plot the selected point
plotacf = inacf.sel(lon=lonf,lat=latf,method='nearest')
classid = int(classmap.sel(lon=lonf,lat=latf,method='nearest').data.item())
cmk   = classmarker[classid]
plotc = classcol[classid]
ax.plot(lags,plotacf,c=plotc,
        ls='dashed',marker=cmk,
        alpha=1,label=expnames_long[ex] + ", %s" % loctitle,lw=4)

# Make Legend and Save    
ax.legend(fontsize=8)
ax.set_title("%s (%s)" % (bbname,bbti))
ax.set_ylabel("Correlation with %s Anomalies" % (mons3[kmonth]))
savename = "%sPointwise_ACF_%s_%s_region_%s_%02i.png" % (figpath,comparename,bbname[:3],expnames[ex],kmonth+1)
plt.savefig(savename,dpi=150)

#%% Plot Monthly Variances

vlm          =[0,0.02] # IRM SSS
#vlm = [0,0.009] #SAR SSS
#vlm = [0,0.045] # NAC SSS
use_class_col=True
ex           = 3


fig,ax = viz.init_monplot(1,1,figsize=(10,6))


inacf = monvar_byexp[ex].mean('ens')

# Plot the individual points
class_avg =[[],[],[],[]]
for a in range(nlat):
    for o in range(nlon):
        plotacf = inacf.isel(lat=a,lon=o)
        
        
        classid = classmap.isel(lon=o,lat=a).data.item()
        
        if np.isnan(classid):
            print("Nan Detected")
            continue
        else:
            classid = int(classid)
            
            
        cmk = classmarker[classid]
        if use_class_col:
            plotc = classcol[classid]
        else:
            plotc = ecols[ex]
            
            
        ax.plot(mons3,plotacf,c=plotc,alpha=0.3,marker=cmk)
        class_avg[classid].append(plotacf)


# Plot Class Averages
for c in range(nclasses):
    plotacf = np.nanmean(np.array(class_avg[c]),0)
    ax.plot(mons3,plotacf,c=classcol[c],alpha=1,marker=classmarker[c],label="%s" % threslabs[c])

# Plot Region Mean
plotacf = inacf.mean('lat').mean('lon')
ax.plot(mons3,plotacf,c=ecols[ex],alpha=1,label=expnames_long[ex] + ", Region Mean ACF",lw=2.5)

# Plot the selected point
plotacf = inacf.sel(lon=lonf,lat=latf,method='nearest')
classid = int(classmap.sel(lon=lonf,lat=latf,method='nearest').data.item())
cmk   = classmarker[classid]
plotc = classcol[classid]
ax.plot(mons3,plotacf,c=plotc,
        ls='dashed',marker=cmk,
        alpha=1,label=expnames_long[ex] + ", %s" % loctitle,lw=4)

ax.set_ylim(vlm)
ax.legend(fontsize=8)
ax.set_title("%s (%s), %s" % (bbname,bbti,expnames_long[ex]))
ax.set_ylabel("Variance [%s]$^2$" % (vunit))
savename = "%sPointwise_Monvar_%s_region_%s_classify_%s.png" % (figpath,comparename,bbname[:3],expnames[ex])
plt.savefig(savename,dpi=150)

#%% Investigate location of points based on month of maximum variance
# Replicate the plot above


ex              = 3
monvar_max      = monvar_byexp[ex].mean('ens').argmax('month') + 1
plot_SST_rei    = True
plotvar         = monvar_max

plotname        = "Month of Max Interannual Variance"
plotname_fn     = "MonvarMax"


# Make the Figure
fig,ax,bbplotr = viz.init_regplot(bbnameshort,)

# Plot Markers of each class
for o in range(nlon):
    for a in range(nlat):
        classpt = classmap.isel(lon=o,lat=a)
        if np.isnan(classpt):
            continue
        cid     = int(classpt.item())
        ax.plot(classpt.lon,classpt.lat,transform=proj,
                marker=classmarker[cid],c=classcol[cid],markersize=15,zorder=2)
        
# Plot Mean Currents
# Plot Currents
plotu = ds_uvel.UVEL.mean('ens').mean('month').values
plotv = ds_vvel.VVEL.mean('ens').mean('month').values
tlon = ds_uvel.TLONG.mean('ens').data
tlat = ds_uvel.TLAT.mean('ens').data
ax.quiver(tlon[::qint, ::qint], tlat[::qint, ::qint], plotu[::qint, ::qint], plotv[::qint, ::qint],
          color='navy', transform=proj, alpha=0.25)

# Plot the Target Variable
pcm         = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,zorder=-1)

# Colorbar Business
cb          = viz.hcbar(pcm,ax=ax,fraction=0.045)
cb.ax.tick_params(labelsize=fsz_tick)
cb.set_label(plotname,fontsize=fsz_axis)

# PLot REI
if varname == "SSS" and plot_SST_rei is False:
    plotvar = reiplot_sss * mask_reg
elif varname == "SST" or plot_SST_rei:
    plotvar = reiplot_sst * mask_reg
    
cl = ax.contour(plotvar.lon,plotvar.lat,plotvar,
                levels=cints_rei,colors="k",lw=2.5,
                transform=proj)

ax.clabel(cl,fontsize=fsz_tick)

# Plot Labels for points
plotvar = proc.sel_region_xr(monvar_max,bbplotr)
for (i, j), z in np.ndenumerate(plotvar):
    ax.text(plotvar.lon.data[j], plotvar.lat.data[i], '%i' % (z),
            ha='center', va='center',transform=proj,fontsize=13,color='lightgray',zorder=4,)#path_effects=[pe.withStroke(linewidth=1.5, foreground="w")])

# Select A Point
reipt = plotvar.sel(lon=lonf,lat=latf,method='nearest')
print(reipt)
ax.plot(reipt.lon,reipt.lat,marker="+",markersize=42,c="k",transform=proj)

savename = "%sLocator_%s_%s_region_%s_MonvarMax.png" % (figpath,plotname_fn,comparename,bbname[:3])
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% For a selected point, visualize the metrics (just monthly variance and acf)


if bbnameshort == "IRM":
    bblocator = [-42,-23,48,62]
elif bbnameshort == "SAR":
    bblocator = [-72,-53,32,42]
else:
    bblocator = bbplotr
    
lonf    = -40
latf    = 50
kmonth  = 1 

plotvar = (monvar_byexp[0].mean('ens') - monvar_byexp[-1].mean('ens')).mean('month')
plotname_choose = 'monvar_overest'


lonr = acfs_byexp[-1].lon.data
latr = acfs_byexp[-1].lat.data

for o in range(nlon):
    for a in range(nlat):
        lonf=lonr[o]
        latf=latr[a]
        
        locfn,loctitle = proc.make_locstring(lonf,latf)
        
        vname   = monvar_byexp[-1].name
        if vname == "SSS":
            vunit = "psu"
            plot_rei   = reiplot_sss
            vlms       = [0,0.030]
            #plotvar    = reiplot_sst
            #plotvarlab = reiplot_sss
            plotname = "REI (colors <%s>, contours <SSS>)" % plotname_choose
        else:
            vunit = "\degree C"
            plot_rei = reiplot_sst
            vlms      = [0,1]
            #plotvar  = reiplot_sss
            #plotvarlab = reiplot_sst
            plotname = "REI (colors <%s>, contours <SST>)" % plotname_choose
        
        
        fig       = plt.figure(figsize=(18,15))
        gs        = gridspec.GridSpec(3,2)
        
        # Locator (with values) ---------------------------------------------------------
        ax1 = fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
        ax1.set_extent(bblocator)
        ax1.coastlines()
        ax  = ax1
        
        # Plot Re-emergence INdex
        cl = ax.contour(plot_rei.lon, plot_rei.lat, plot_rei * mask,
                    colors="k", transform=proj, zorder=2)
        ax.clabel(cl)
        
        # PLot Location
        ax.plot(lonf,latf,marker="x",markersize=25,c="k")
        
        # # Plot Markers of each class
        # for o in range(nlon):
        #     for a in range(nlat):
        #         classpt = classmap.isel(lon=o,lat=a)
        #         if np.isnan(classpt):
        #             continue
        #         cid     = int(classpt.item())
        #         ax.plot(classpt.lon,classpt.lat,transform=proj,
        #                 marker=classmarker[cid],c=classcol[cid],markersize=15,zorder=2)
        
        
        # Plot Currents
        plotu = ds_uvel.UVEL.mean('ens').mean('month').values
        plotv = ds_vvel.VVEL.mean('ens').mean('month').values
        tlon = ds_uvel.TLONG.mean('ens').data
        tlat = ds_uvel.TLAT.mean('ens').data
        ax.quiver(tlon[::qint, ::qint], tlat[::qint, ::qint], plotu[::qint, ::qint], plotv[::qint, ::qint],
                  color='navy', transform=proj, alpha=0.4)
        
        # Plot the Target Variable
        pcm         = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,transform=proj,zorder=-1)
        
        # Colorbar Business
        cb          = fig.colorbar(pcm,ax=ax,fraction=0.02,pad=0.01)
        cb.ax.tick_params(labelsize=fsz_tick)
        #cb.set_label(plotname,fontsize=fsz_axis)
        ax.set_title(plotname,fontsize=fsz_axis)
        
        
        # Plot Labels for points
        plotvar = proc.sel_region_xr(plotvar,bblocator)
        for (i, j), z in np.ndenumerate(plotvar):
            ax.text(plotvar.lon.data[j], plotvar.lat.data[i], '{:0.2f}'.format(z),
                    ha='center', va='center',transform=proj,fontsize=9,color='lightgray',zorder=4,)#path_effects=[pe.withStroke(linewidth=1.5, foreground="w")])
        
        
        # # Plot Monthly Variance  ------------------------------------------------------
        ax2 = fig.add_subplot(gs[0,1])
        ax          = ax2
        ax          = viz.format_monplot(ax)
        for ex in range(nexps):
            inmonvar    = monvar_byexp[ex].mean('ens').sel(lon=lonf,lat=latf,method='nearest')
            ax.plot(mons3,inmonvar,c=ecols[ex],label=expnames_long[ex])
        ax.legend(ncol=2)
        ax.set_ylim(vlms)
        ax.set_title("%s Interannual Variability by Month" % (vname))
        
        #fig,axs = plt.subplots(1,2,constrained_layout=True,figsize=(12,4.5))
        
        # # Plot ACF --------------------------------------------------------------------
        ax3 = fig.add_subplot(gs[1,:])
        ax = ax3
        ax,_    = viz.init_acplot(kmonth,xtks,lags,title="",ax=ax)
        for ex in range(nexps):
            inacf  = acfs_byexp[ex].isel(basemon=kmonth).sel(lon=lonf,lat=latf,method='nearest')
            if 'ens' in list(inacf.dims):
                #print(ex)
                inacf = inacf.mean('ens')
            ax.plot(lags,inacf,c=ecols[ex],label=expnames_long[ex])
        ax.legend(ncol=2)
        #ax.set_title("%s %s ACF" % (vname,mons3[kmonth]))
        
        
        plt.suptitle("Location: %s" % (loctitle),fontsize=fsz_title,y=0.92)
        
        
        
        savename = "%s%sMonvar_and_ACF_mon%02i_%s.png" % (figpath,comparename,kmonth,locfn)
        plt.savefig(savename,dpi=150,bbox_inches='tight')
        


#%% Investigate Intraregional Differences

if bbnameshort == "IRM":
    figsize = (12,10)
    lonf = -35
    latf = 53
elif bbnameshort == "SAR":
    figsize = (12,8)
    lonf = -65
    latf = 36
else:
    figsize = (12,10)
    
    
if vname == "SSS":
    
    pvlim = [-.5,.5]
    mvlim = [-.008,0.008]
    
elif vname == "SST":
    
    pvlim = [-.1,.1]
    mvlim = [-.1,.1]
    
    

fig,axs = plt.subplots(2,2,subplot_kw={'projection':ccrs.PlateCarree()},figsize=figsize,constrained_layout=True)

for ax in axs.flatten():
    #ax.coastlines()
    #ax.set_extent(bblocator)
    ax = viz.add_coast_grid(ax,bblocator)
    
    plotu = ds_uvel.UVEL.mean('ens').mean('month').values
    plotv = ds_vvel.VVEL.mean('ens').mean('month').values
    tlon = ds_uvel.TLONG.mean('ens').data
    tlat = ds_uvel.TLAT.mean('ens').data
    
    ax.quiver(tlon[::qint, ::qint], tlat[::qint, ::qint], plotu[::qint, ::qint], plotv[::qint, ::qint],
              color='navy', transform=proj, alpha=0.75,zorder=3)
    
    
    # Plot Ice Edge
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2.5,
               transform=proj,levels=[0,1],zorder=-1)
    
    # Plot Marker
    ax.plot(lonf,latf,c='yellow',markersize=25,marker="x")
    #viz.plot_box()
    
# REI Index (Salinity) --------------------------------------------------------
ax      = axs[0,0]
plotname = "REI (SSS)"
plotvar = reiplot_sss * mask_reg
plotvar = proc.sel_region_xr(plotvar,bblocator)
pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,cmap='cmo.deep',vmin=0,vmax=0.55,)
# Plot Contour
cl = ax.contour(plotvar.lon, plotvar.lat, plotvar * mask,
            colors="k", transform=proj, zorder=2)
ax.clabel(cl)
cb  = viz.hcbar(pcm,ax=ax,fraction=0.052)
cb.set_label(plotname,fontsize=fsz_axis)
# Plot values
for (i, j), z in np.ndenumerate(plotvar):
    ax.text(plotvar.lon.data[j], plotvar.lat.data[i], '{:0.2f}'.format(z),
            ha='center', va='center',transform=proj,fontsize=9,color='gray',zorder=4,)#path_effects=[pe.withStroke(linewidth=1.5, foreground="w")])


# REI Index (SST) --------------------------------------------------------
ax      = axs[0,1]
plotname = "REI (SST)"
plotvar = reiplot_sst * mask_reg
plotvar = proc.sel_region_xr(plotvar,bblocator)
pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,cmap='cmo.dense',vmin=0,vmax=0.55,)
# Plot Contour
cl = ax.contour(plotvar.lon, plotvar.lat, plotvar * mask_reg,
            colors="k", transform=proj, zorder=2)
ax.clabel(cl)
cb  = viz.hcbar(pcm,ax=ax,fraction=0.052)
cb.set_label(plotname,fontsize=fsz_axis)
# Plot values
for (i, j), z in np.ndenumerate(plotvar):
    ax.text(plotvar.lon.data[j], plotvar.lat.data[i], '{:0.2f}'.format(z),
            ha='center', va='center',transform=proj,fontsize=9,color='lightgray',zorder=4,)#path_effects=[pe.withStroke(linewidth=1.5, foreground="w")])


# Overestimate of Persistence --------------------------------------------------------
ax      = axs[1,0]
plotname = "%s Persistence Overestimate (SM - CESM1)" % (mons3[kmonth])
plotvar  = (acfs_byexp[0].isel(basemon=kmonth) - acfs_byexp[-1].mean('ens').isel(basemon=kmonth)).mean('lags').T *  mask
plotvar = proc.sel_region_xr(plotvar,bblocator)
pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,cmap='cmo.balance',vmin=pvlim[0],vmax=pvlim[1])

# Plot Contour
cl = ax.contour(plotvar.lon, plotvar.lat, plotvar * mask,
            colors="k", transform=proj, zorder=2)
ax.clabel(cl)
cb  = viz.hcbar(pcm,ax=ax,fraction=0.052)
cb.set_label(plotname,fontsize=fsz_axis)

# Plot values
for (i, j), z in np.ndenumerate(plotvar):
    ax.text(plotvar.lon.data[j], plotvar.lat.data[i], '{:0.2f}'.format(z),
            ha='center', va='center',transform=proj,fontsize=9,color='lightgray',zorder=4,)#path_effects=[pe.withStroke(linewidth=1.5, foreground="w")])


# (Over)Estimation of Monthly Variance ----------------------------------------
ax      = axs[1,1]
plotname = "%s Monvar Overestimate (SM - CESM1)" % (vname)
plotvar  = (monvar_byexp[0].mean('ens') - monvar_byexp[-1].mean('ens')).mean('month') *  mask
plotvar = proc.sel_region_xr(plotvar,bblocator)
pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,cmap='cmo.balance',vmin=mvlim[0],vmax=mvlim[1])

# Plot Contour
cl = ax.contour(plotvar.lon, plotvar.lat, plotvar * mask,
            colors="k", transform=proj, zorder=2)
ax.clabel(cl)
cb  = viz.hcbar(pcm,ax=ax,fraction=0.052)
cb.set_label(plotname,fontsize=fsz_axis)

# Plot values
for (i, j), z in np.ndenumerate(plotvar):
    ax.text(plotvar.lon.data[j], plotvar.lat.data[i], '{:0.2f}'.format(z),
            ha='center', va='center',transform=proj,fontsize=9,color='lightgray',zorder=4,)#path_effects=[pe.withStroke(linewidth=1.5, foreground="w")])

plt.suptitle("Regional Consistency For %s (Exp: %s)" % (bbnameshort,comparename),fontsize=fsz_title,y=1.05)

savename = "%s%s_%s_Regional_consistency_check_mon%02i.png" % (figpath,bbnameshort,comparename,kmonth)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% CHECK MONTHS WHERE MAXIMUM OCURr



if bbnameshort == "IRM":
    figsize = (12,10)
    lonf = -35
    latf = 53
elif bbnameshort == "SAR":
    figsize = (12,8)
    lonf = -65
    latf = 36
else:
    figsize = (12,10)
    
    
if vname == "SSS":
    
    pvlim = [-.5,.5]
    mvlim = [-.008,0.008]
    
elif vname == "SST":
    
    pvlim = [-.1,.1]
    mvlim = [-.1,.1]
    
    

fig,axs = plt.subplots(2,2,subplot_kw={'projection':ccrs.PlateCarree()},figsize=figsize,constrained_layout=True)

for ax in axs.flatten():
    #ax.coastlines()
    #ax.set_extent(bblocator)
    ax = viz.add_coast_grid(ax,bblocator)
    
    plotu = ds_uvel.UVEL.mean('ens').mean('month').values
    plotv = ds_vvel.VVEL.mean('ens').mean('month').values
    tlon = ds_uvel.TLONG.mean('ens').data
    tlat = ds_uvel.TLAT.mean('ens').data
    
    ax.quiver(tlon[::qint, ::qint], tlat[::qint, ::qint], plotu[::qint, ::qint], plotv[::qint, ::qint],
              color='navy', transform=proj, alpha=0.75,zorder=3)
    
    
    # Plot Ice Edge
    ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2.5,
               transform=proj,levels=[0,1],zorder=-1)
    
    # Plot Marker
    ax.plot(lonf,latf,c='yellow',markersize=25,marker="x")
    #viz.plot_box()
    
# REI Index (Salinity) --------------------------------------------------------
ax      = axs[0,0]
plotname = "REI (SSS)"
plotvar = reiplot_sss * mask_reg
plotvar = proc.sel_region_xr(plotvar,bblocator)
pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,cmap='cmo.deep',vmin=0,vmax=0.55,)
# Plot Contour
cl = ax.contour(plotvar.lon, plotvar.lat, plotvar * mask,
            colors="k", transform=proj, zorder=2)
ax.clabel(cl)
cb  = viz.hcbar(pcm,ax=ax,fraction=0.052)
cb.set_label(plotname,fontsize=fsz_axis)
# Plot values
for (i, j), z in np.ndenumerate(plotvar):
    ax.text(plotvar.lon.data[j], plotvar.lat.data[i], '{:0.2f}'.format(z),
            ha='center', va='center',transform=proj,fontsize=9,color='gray',zorder=4,)#path_effects=[pe.withStroke(linewidth=1.5, foreground="w")])


# REI Index (SST) --------------------------------------------------------
ax      = axs[0,1]
plotname = "REI (SST)"
plotvar = reiplot_sst * mask_reg
plotvar = proc.sel_region_xr(plotvar,bblocator)
pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,cmap='cmo.dense',vmin=0,vmax=0.55,)
# Plot Contour
cl = ax.contour(plotvar.lon, plotvar.lat, plotvar * mask_reg,
            colors="k", transform=proj, zorder=2)
ax.clabel(cl)
cb  = viz.hcbar(pcm,ax=ax,fraction=0.052)
cb.set_label(plotname,fontsize=fsz_axis)
# Plot values
for (i, j), z in np.ndenumerate(plotvar):
    ax.text(plotvar.lon.data[j], plotvar.lat.data[i], '{:0.2f}'.format(z),
            ha='center', va='center',transform=proj,fontsize=9,color='lightgray',zorder=4,)#path_effects=[pe.withStroke(linewidth=1.5, foreground="w")])


# Overestimate of Persistence --------------------------------------------------------
ax      = axs[1,0]
plotname = "Max Interannual %s Variance" % (vname)
plotvar  = ((monvar_byexp[-1].mean('ens')).min('month')) * mask
plotvar = proc.sel_region_xr(plotvar,bblocator)
pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,cmap='cmo.thermal',vmin=0,vmax=1)

# Plot Contour
# cl = ax.contour(plotvar.lon, plotvar.lat, plotvar * mask,
#             colors="k", transform=proj, zorder=2)
# ax.clabel(cl)
cb  = viz.hcbar(pcm,ax=ax,fraction=0.052)
cb.set_label(plotname,fontsize=fsz_axis)

# Plot values
for (i, j), z in np.ndenumerate(plotvar):
    ax.text(plotvar.lon.data[j], plotvar.lat.data[i], '%.2f' % z,
            ha='center', va='center',transform=proj,fontsize=9,color='lightgray',zorder=4,)#path_effects=[pe.withStroke(linewidth=1.5, foreground="w")])

# (Over)Estimation of Monthly Variance ----------------------------------------
ax      = axs[1,1]
plotname = "Month of Max Interannual %s Variance" % (vname)
plotvar  = ((monvar_byexp[-1].mean('ens')).argmin('month') +1 ) * mask
plotvar = proc.sel_region_xr(plotvar,bblocator)
pcm = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,cmap='twilight',vmin=1,vmax=12)

# Plot Contour
# cl = ax.contour(plotvar.lon, plotvar.lat, plotvar * mask,
#             colors="k", transform=proj, zorder=2)
# ax.clabel(cl)
cb  = viz.hcbar(pcm,ax=ax,fraction=0.052)
cb.set_label(plotname,fontsize=fsz_axis)

# Plot values
for (i, j), z in np.ndenumerate(plotvar):
    ax.text(plotvar.lon.data[j], plotvar.lat.data[i], '%i' % z,
            ha='center', va='center',transform=proj,fontsize=9,color='lightgray',zorder=4,)#path_effects=[pe.withStroke(linewidth=1.5, foreground="w")])

plt.suptitle("Regional Consistency For %s (Exp: %s)" % (bbnameshort,comparename),fontsize=fsz_title,y=1.05)

savename = "%s%s_%s_Regional_consistency_monmax_check_mon%02i.png" % (figpath,bbnameshort,comparename,kmonth)
plt.savefig(savename,dpi=150,bbox_inches='tight')
#%%






# #%%
# for kmonth in range(12):
#     if kmonth not in loopmon:
#         continue
    
    
#     locfn,loctitle=proc.make_locstring(lonf,latf)
    
#     lags = np.arange(nlags)
#     xtks = lags[::2]
    
#     fig, ax = plt.subplots(1, 1,constrained_layout=True,figsize=(10,6))
#     ax, _ = viz.init_acplot(kmonth, xtks, lags, title=None)
    
#     for ex in range(nexps):
#         if ex not in plotex:
#             continue
        
#         inacf = acfs_byexp[ex].isel(basemon=kmonth)#.mean('ens')
#         if 'ens' in list(inacf.dims):
#             inacf = inacf.mean('ens')
            
#         # Plot the individual points
#         for a in range(nlat):
#             for o in range(nlon):
                
#                 classid = int(classmap.isel(lon=o,lat=a).data.item())
#                 cmk = classmarker[classid]
#                 if use_class_col:
#                     plotc = classcol[classid]
#                 else:
#                     plotc = ecols[ex]
                
#                 plotacf= inacf.isel(lat=a,lon=o)
#                 ax.plot(lags,plotacf,c=plotc,alpha=0.3,marker=cmk)
        
#         # Plot Region Mean
#         plotacf = inacf.mean('lat').mean('lon')
#         ax.plot(lags,plotacf,c=ecols[ex],alpha=1,label=expnames_long[ex] + ", Region Mean ACF",lw=2.5)
        
#         # Plot the selected point
#         plotacf = inacf.sel(lon=lonf,lat=latf,method='nearest')
#         ax.plot(lags,plotacf,c=ecols[ex],
#                 ls='dashed',marker="x",
#                 alpha=1,label=expnames_long[ex] + ", %s" % loctitle,lw=2.5)
        
        
#     ax.legend(fontsize=8)
#     ax.set_title("%s (%s)" % (bbname,bbti))
#     ax.set_ylabel("Correlation with %s Anomalies" % (mons3[kmonth]))
#     savename = "%sPointwise_ACF_%s_region_%s_%02i.png" % (figpath,comparename,bbname[:3],kmonth+1)
#     plt.savefig(savename,dpi=150)
#%% Overestimates in monthly variance
        
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
savename = "%sPointwise_Monvar_%s_region_%s_classify.png" % (figpath,comparename,bbname[:3])
plt.savefig(savename,dpi=150)





#%% Now try to plot ACFs with the regions



# # Initialize Plot and Map
# fig, ax, _ = viz.init_orthomap(1, 1, bboxplot, figsize=(18, 6.5))
# ax         = viz.add_coast_grid(ax, bbox=bboxplot, fill_color="lightgray")


# for c in range(nclasses):
#     classpoints = xr.where(classmap==c,c,)
    
#     ax.pcolormesh(classpoints.lon,classpoints.lat,classpoints,transform=True,
#             color=classcol[c],
#             marker=classmarker[c])
    



