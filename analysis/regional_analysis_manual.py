#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Manually do some regional analysis (ACF average)

Created on Thu Jul  4 15:22:32 2024

@author: gliu


"""

import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs

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

#%%

bboxplot                        = [-80,0,20,65]
mpl.rcParams['font.family']     = 'Avenir'
mons3                           = proc.get_monstr(nletters=3)

fsz_tick                        = 18
fsz_axis                        = 20
fsz_title                       = 16

rhocrit                         = proc.ttest_rho(0.05,2,86)

proj                            = ccrs.PlateCarree()


#%%  Indicate Experients (copying upper setion of viz_regional_spectra )


# #  Same as comparing lbd_e effect, but with Evaporation forcing corrections
#regionset       = "SSSCSU"
comparename     = "SSS_Paper_Draft01"
expnames        = ["SSS_EOF_LbddCorr_Rerun_lbdE_neg","SSS_EOF_LbddCorr_Rerun","SSS_EOF_LbddCorr_Rerun_NoLbdd","SSS_CESM"]
expnames_long   = ["Stochastic Model (sign corrected + $\lambda^e$)","Stochastic Model (with $\lambda^e$)","Stochastic Model","CESM1"]
expnames_short  = ["SM_lbde_neg","SM_lbde","SM","CESM"]
ecols           = ["magenta","forestgreen","goldenrod","k"]
els             = ['dotted',"solid",'dashed','solid']
emarkers        = ['+',"d","x","o"]



cesm_exps = ["SST_CESM","SSS_CESM"]


#%% Load the Dataset (us sm output loader)
# Hopefully this doesn't clog up the memory too much

nexps = len(expnames)
ds_all = []
for e in tqdm.tqdm(range(nexps)):
    
    # Get Experiment information
    expname        = expnames[e]
    
    if "SSS" in expname:
        varname = "SSS"
    elif "SST" in expname:
        varname = "SST"
    
    # For stochastic model output
    ds = dl.load_smoutput(expname,output_path)
    
    if expname in cesm_exps:
        print("Detrending and deseasoning")
        ds = proc.xrdeseason(ds[varname])
        ds = ds - ds.mean('ens')
        ds = xr.where(np.isnan(ds),0,ds) # Sub with zeros for now
    else:
        ds = ds[varname]
        
    ds_all.append(ds)
        

#%% Detrend the cesm1 input

#%% Load some variables to plot 


# Load Current
ds_uvel,ds_vvel = dl.load_current()

# load SSS Re-emergence index (for background plot)
ds_rei = dl.load_rei("SSS_CESM",output_path).load().rei


# Load Gulf Stream
ds_gs = dl.load_gs()
ds_gs = ds_gs.sel(lon=slice(-90,-50))


#%% Select a region (Sargasso Sea Adjustment)

#%% Plot Locator and Bounding Box w.r.t. the currents

#sel_box   = [-70,-55,35,40]
sel_box   = [-58,-48,55,60]
qint      = 2

# Restrict REI for plotting
selmons   = [1,2]
iyr       = 0
plot_rei  = ds_rei.isel(mon=selmons,yr=iyr).mean('mon').mean('ens')
rei_cints = np.arange(0,0.55,0.05)
rei_cmap  = 'cmo.deep' 

# Initialize Plot and Map
fig,ax,_    = viz.init_orthomap(1,1,bboxplot,figsize=(18,6.5))
ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
ax.set_title("CESM1 Historical Ens. Avg., Ann. Mean",fontsize=fsz_title)

# Plot Currents
plotu = ds_uvel.UVEL.mean('ens').mean('month').values
plotv = ds_vvel.VVEL.mean('ens').mean('month').values
tlon = ds_uvel.TLONG.mean('ens').data
tlat = ds_uvel.TLAT.mean('ens').data

ax.quiver(tlon[::qint,::qint],tlat[::qint,::qint],plotu[::qint,::qint],plotv[::qint,::qint],
          color='navy',transform=proj,alpha=0.75)

l1 = viz.plot_box(sel_box)

# Plot Re-emergence INdex
ax.contourf(plot_rei.lon,plot_rei.lat,plot_rei,cmap='cmo.deep',transform=proj,zorder=-1)


# Plot Gulf Stream Position
ax.plot(ds_gs.lon,ds_gs.lat.mean('ens'),transform=proj,lw=1.75,c="k")


#%% Perform Regional Subsetting and Analysis

dsreg     = [proc.sel_region_xr(ds,sel_box) for ds in ds_all]
regavg_ts = [ds.mean('lat').mean('lon').data for ds in dsreg]

tsm_byexp = []
for e in range(nexps):
    ts_in   = regavg_ts[e]
    nrun    = ts_in.shape[0]
    print(nrun)
    
    ts_list = [ts_in[ii,:] for ii in range(nrun)] 
    print(ts_in.shape)
    
    tsm     = scm.compute_sm_metrics(ts_list)
    
    tsm_byexp.append(tsm)


print(tsm.keys())

#%% Visualzie the regional ACF

lags    = np.arange(37)
xtks    = lags
kmonth  = 1

fig,ax= plt.subplots(1,1,constrained_layout=True,figsize=(12,4.5))
ax,_  = viz.init_acplot(kmonth,xtks,lags,ax=ax,title="")

for ex in range(nexps):
    
    acfexp = np.array(tsm_byexp[ex]['acfs'][kmonth]) # Run x Lag
    
    ax.plot(lags,acfexp.mean(0),label=expnames_long[ex],
            c=ecols[ex],ls=els[ex])
    

    
    

#%% ===========================================================================
#%% ===========================================================================

# #%% Old Version, focused on using ACFs
    
# # CESM2 vs. CESM1 Stochastic Model
# compare_name = "cesm2vcesm1_picsm"

# ncname = "cesm2_pic_0200to2000_TS_ACF_lag00to60_ALL_ensALL.nc"
# vname  = "TS"
# ncpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
# ds1    = xr.open_dataset(ncpath+ncname).acf.load().squeeze()


# ncname = "SM_SST_cesm2_pic_noQek_SST_autocorrelation_thresALL_lag00to60.nc"
# vname  = "SST"
# ncpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
# ds2    = xr.open_dataset(ncpath+ncname)[vname].load()#.acf.load().squeeze()

# ds_in    = [ds1,ds2]
# ds_in    = proc.resize_ds(ds_in)
# expnames = ["CESM2 PIC","Stochastic Model"]

# #%% Compare wintertime ACF over bounding box
# bbsel   = [-45,-30,50,65]
# selmons = [1,]

# ds_sel  = [proc.sel_region_xr(ds.isel(mons=selmons).mean('mons'),bbsel).mean('lat').mean('lon') for ds in ds_in]

# #%% Plot Mean ACF


# kmonth    = selmons[0]
# lags      = ds_sel[0].lags.data
# xtks      = lags[::3]


# fig       = plt.figure(figsize=(18,6.5))
# gs        = gridspec.GridSpec(4,4)


# # --------------------------------- # Locator
# ax1       = fig.add_subplot(gs[0:3,0],projection=ccrs.PlateCarree())
# ax1       = viz.add_coast_grid(ax1,bbox=bboxplot,fill_color="lightgray")
# ax1.set_title(bbsel)
# ax1 = viz.plot_box(bbsel)



# ax2       = fig.add_subplot(gs[1:3,1:])
# ax2,_     = viz.init_acplot(kmonth,xtks,lags,title="",)


# for ii in range(2):
#     ax2.plot(lags,ds_sel[ii].squeeze().data,label=expnames[ii])

# ax2.legend()

# #%% Section here is copied from regional spectral analysis




