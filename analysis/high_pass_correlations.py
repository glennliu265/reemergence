#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


Examine how applying high pass filters impacts the correlation of SST and SSS

Created on Wed May 29 15:11:59 2024

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
machine = "stormtrack"

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


#%% Load SST and SSS



datname = "CESM1_LE"
vnames  = ["TEMP","SSS"]
ncnames = ["CESM1LE_TEMP_NAtl_19200101_20050101_NN.nc","CESM1LE_SSS_NAtl_19200101_20050101_bilinear.nc"]


# datname = "SM_Default"
# vnames  = ["SST","SSS"]
# ncnames = ["SST_EOF_LbddCorr_Rerun","SSS_EOF_LbddCorr_Rerun_lbdE_neg"]



datname = "SM_LHFLX"
vnames  = ["SST","SSS"]
ncnames = ["SST_EOF_LHFLX","SSS_EOF_LHFLX_lbdE"]


# nc_base      = "SST_EOF_LbddCorr_Rerun" # [ensemble x time x lat x lon 180]
# nc_lag       = "SSS_EOF_LbddCorr_Rerun_lbdE_neg" # [ensemble x time x lat x lon 180]


save_hpfoutput = False

#%% Functions
def load_smoutput(expname,output_path,debug=True): # Copied from pointwise_crosscorr
    # Load NC Files
    expdir       = output_path + expname + "/Output/"
    nclist       = glob.glob(expdir +"*.nc")
    nclist.sort()
    if debug:
        print(nclist)
        
    # Load DS, deseason and detrend to be sure
    ds_all   = xr.open_mfdataset(nclist,concat_dim="run",combine='nested').load()
    return ds_all

def preproc_ds(ds):
    if 'ensemble' in list(ds.dims):
        ds=ds.rename({'ensemble':'ens'})
    if 'run' in list(ds.dims):
        ds=ds.rename({'run':'ens'})
    dsa = proc.xrdeseason(ds)#ds - ds.groupby('time.month').mean('time')
    dsa = dsa - dsa.mean('ens')
    return dsa

#%% Load the data

st = time.time()
if "SM_" in datname:
    print("Stochastic Model Output Detected")
    ds_load = []
    for ii in range(2):
        dsvar = load_smoutput(ncnames[ii],output_path)
        ds_load.append(dsvar[vnames[ii]])
        #smpath = output_path + ncnames[ii] + "/Output/"
        #nclist = 
else:
    ds_load = [xr.open_dataset(rawpath+ncnames[ii])[vnames[ii]].load() for ii in range(2)]

print("Loaded in %.2fs" % (time.time()-st))

#%% Detrend and deseason

st = time.time()
ds_anom = [preproc_ds(ds) for ds in ds_load]
print("Preproc in %.2fs" % (time.time()-st))

#%% Design and apply pointwise high pass filters (all months, loop ver)
# Based on script in viz_SST_SSS_coupling

#% Single cutoff

hicutoff  = 12

hipass    = lambda x: proc.lp_butter(x,hicutoff,6,btype='highpass') # Make Function
cesm_hipass = []
for vv in tqdm.tqdm(range(2)):
    hpout = xr.apply_ufunc(
        hipass,
        ds_anom[vv],
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        vectorize=True, 
        )
    cesm_hipass.append(hpout)
# Takes 15 minutes per iteration ()

# Save outut
if save_hpfoutput:
    for vv in range(2):
        hipass_out = cesm_hipass[vv]
        outname    = rawpath + "/filtered/" + proc.addstrtoext(ncnames[vv],"_hpf%02imon" % hicutoff,adjust=-1)
        hipass_out.to_netcdf(outname,encoding={vnames[vv]:{'zlib':True}})
    
if "CESM1" in datname:
    cesm_hipass[1]['ens'] = np.arange(1,43,1)
    ds_anom[1]['ens'] = np.arange(1,43,1)
# Resize so they are the same size
cesm_rsz = proc.resize_ds(cesm_hipass)
dsanom_rsz = proc.resize_ds(ds_anom)

# Get the cross correlation
crosscorr = lambda x,y: np.corrcoef(x,y)[0,1]
ccout = xr.apply_ufunc(
    crosscorr,
    cesm_rsz[0],
    cesm_rsz[1],
    input_core_dims=[['time'],['time']],
    output_core_dims=[[]],
    vectorize=True, 
    )



# Save output
outname    = rawpath + "/filtered/" + "%s_SST_SSS_NATL_crosscorr_hpf%02imon.nc" % (datname,hicutoff)
ccout      =  ccout.rename("corr")
ccout.to_netcdf(outname,encoding={'corr':{'zlib':True}})
print("File saved to %s" % outname)


# Redo the above with no high-pass correlation
ccout_raw = xr.apply_ufunc(
    crosscorr,
    dsanom_rsz[0],
    dsanom_rsz[1],
    input_core_dims=[['time'],['time']],
    output_core_dims=[[]],
    vectorize=True, 
    )
outname    = rawpath + "/filtered/" + "%s_SST_SSS_NATL_crosscorr_raw.nc" % (datname)
ccout_raw      =  ccout_raw.rename("corr")
ccout_raw.to_netcdf(outname,encoding={'corr':{'zlib':True}})
print("Raw File saved to %s" % outname)

# %% End Official Script (Working Section)

#%%
# Debug by checking at 1 point
lonf = -30
latf = 50

ccpt = proc.selpt_ds(ccout,lonf,latf)

varspt = [proc.selpt_ds(ds,lonf,latf) for ds in cesm_hipass]

testcc =  np.corrcoef(varspt[0].isel(month=1,ens=1).values,varspt[1].isel(month=1,ens=1).values)[0,1]

#%% Above, but loop ver (CESM Support Only)

hicutoffs   = [3,6,9,12,15,18,24] # In Months
nthres      = len(hicutoffs)

hpvars      = []
hpcorr      = []
for th in range(nthres):
    
    # Apply High Pass Filter to each variable
    hicutoff = hicutoffs[th]
    hipass    = lambda x: proc.lp_butter(x,hicutoff,6,btype='highpass') # Make Function
    cesm_hipass = []
    for vv in tqdm.tqdm(range(2)):
        hpout = xr.apply_ufunc(
            hipass,
            ds_anom[vv],
            input_core_dims=[['time']],
            output_core_dims=[['time']],
            vectorize=True, 
            )
        cesm_hipass.append(hpout)
    #hpvars.append(cesm_hipass)
    # Takes 15 minutes per iteration ()
    

    
    
    cesm_hipass[1]['ens'] = np.arange(1,43,1)

    # Resize so they are the same size
    cesm_rsz = proc.resize_ds(cesm_hipass)
    
    # Save outut
    for vv in range(2):
        hipass_out = cesm_hipass[vv]
        outname    = rawpath + "/filtered/" + proc.addstrtoext(ncnames[vv],"_hpf%02imon" % hicutoff,adjust=-1)
        hipass_out.to_netcdf(outname,encoding={vnames[vv]:{'zlib':True}})

    # Get the cross correlation --------------------
    crosscorr = lambda x,y: np.corrcoef(x,y)[0,1]
    ccout = xr.apply_ufunc(
        crosscorr,
        cesm_rsz[0],
        cesm_rsz[1],
        input_core_dims=[['time'],['time']],
        output_core_dims=[[]],
        vectorize=True, 
        )
    
    # Save output
    outname    = rawpath + "/filtered/" + "CESM1_HTR_SST_SSS_NATL_crosscorr_hpf%02imon.nc" % hicutoff
    ccout      =  ccout.rename("corr")
    ccout.to_netcdf(outname,encoding={'corr':{'zlib':True}})
    
    #cesm_cc.append(ccout)
    #hpcorr.append(cesm_cc)

#%% Visualize the output (Note, I have moved this to astraeus, so need to rerun upper section on diff machine)


#outpath    = rawpath + "/filtered/"
hicutoffs  = [3,6,9,12,15,18,24] # In Months


ds_all = []
ncuts  = len(hicutoffs)

for nn in range(ncuts):
    hicutoff   = hicutoffs[nn]
    outname    = rawpath + "/filtered/" + "CESM1_HTR_SST_SSS_NATL_crosscorr_hpf%02imon.nc" % hicutoff
    ds_all.append(xr.open_dataset(outname).load())
    
    
#%% Plot the cross correlation for each cutoff

#im = 

bboxplot                    = [-80,0,20,65]
mpl.rcParams['font.family'] = 'Avenir'
mons3                       = proc.get_monstr(nletters=3)
fsz_tick                    = 18
fsz_axis                    = 14
fsz_title                   = 16
proj                        = ccrs.PlateCarree()

clvls = np.arange(-1,1.1,.1)


for nn in range(ncuts):
    hicutoff    = hicutoffs[nn]
    fig,ax,_    = viz.init_orthomap(1,1,bboxplot,figsize=(12,4))
    ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray")
    ax.set_title("High pass filter (%02i Months)" % (hicutoff),fontsize=fsz_title)
    
    
    pv = ds_all[nn].corr.mean('ens')#.isel(month=1)#.mean('month')
    
    if "month" in list(pv.dims): # Month Dimension is a mistake, just overwrite it...
        print("Removing Month Dim for hpf %i" % hicutoff)
        pv = pv.isel(month=im)
    cf = ax.contourf(pv.lon,pv.lat,pv.values,transform=proj,levels=clvls,cmap="RdBu_r")
    cb = viz.hcbar(cf,ax=ax,fraction=0.045)
    cb.set_label("SST-SSS Cross-Correlation (Instantaneous)")
    
    plt.savefig("%sSST_SSS_Croscorr_CESM1_hpf%02i_month%02i.png" % (figpath,hicutoff,im+1),bbox_inches='tight')
    
    
    
    



