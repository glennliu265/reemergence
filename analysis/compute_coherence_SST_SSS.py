#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute the Coherence with SST/SSS

Copied upper section of compare_currents


Created on Wed Jul 31 15:49:09 2024

@author: gliu

"""

import xarray as xr
import numpy as np
import matplotlib as mpl
import scipy as sp

import matplotlib.pyplot as plt
import sys
import glob
import os

import tqdm
import time

import cartopy.crs as ccrs

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
pathdict    = rparams.machine_paths[machine]

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

import yo_box as yo

#%% Functions

def chk_dims(ds):
    if 'ensemble' in list(ds.dims):
        print("Renaming Ensemble Dimension --> Ens")
        ds = ds.rename(dict(ensemble='ens'))
    
    if 0 in ds.ens or 42 not in ds.ens:
        print("Renumbering ENS from 1 to 42")
        ds['ens'] = np.arange(1,43,1)
        
    return ds


def preproc_ds(ds):
    dsa = proc.xrdeseason(ds)#ds - ds.groupby('time.month').mean('time')
    dsa = dsa - dsa.mean('ens')
    return dsa

#%% Plotting Information

bboxplot                    = [-80,0,20,65]
mpl.rcParams['font.family'] = 'Avenir'
mons3                       = proc.get_monstr(nletters=3)
fsz_tick                    = 18
fsz_axis                    = 22
fsz_title                   = 28
proj                        = ccrs.PlateCarree()



#%% Load the currents

# Load the geostrophic currents
st          = time.time()
nc_ugeo     = "CESM1LE_ugeo_NAtl_19200101_20050101_bilinear.nc"
path_ugeo   = rawpath
ds_ugeo     = xr.open_dataset(path_ugeo + nc_ugeo).load()
print("Loaded ugeo in %.2fs" % (time.time()-st))

ugeo_mod    = (ds_ugeo.ug **2 + ds_ugeo.vg ** 2) ** 0.5


# # Load the ekman currents
# st          = time.time()
# nc_uek      = "CESM1LE_uek_NAtl_19200101_20050101_bilinear.nc"
# ds_uek      = xr.open_dataset(path_ugeo + nc_uek).load()
# print("Loaded uek in %.2fs" % (time.time()-st))

# # Load total surface velocity
# st          = time.time()
# nc_uvel     = "UVEL_NATL_AllEns_regridNN.nc"
# nc_vvel     = "VVEL_NATL_AllEns_regridNN.nc"
# ds_uvel     = xr.open_dataset(path_ugeo + nc_uvel).load()
# ds_vvel     = xr.open_dataset(path_ugeo + nc_vvel).load()
# print("Loaded uvels in %.2fs" % (time.time()-st))

#%% Load SST/SSS

st     = time.time()
nc_sst = "CESM1LE_SST_NAtl_19200101_20050101_bilinear.nc"
nc_sss = "CESM1LE_SSS_NAtl_19200101_20050101_bilinear.nc"
ds_sst = xr.open_dataset(path_ugeo + nc_sst).SST.load()
ds_sss = xr.open_dataset(path_ugeo + nc_sss).SSS.load()
print("Loaded SST and SSS in %.2fs" % (time.time()-st))

#%% Indicate if it is pointwiwse (set bbox to None) or regional analysis

lonf           = -30
latf           = 50
bbox_sel       = None
bbox_name      = None

#bbox_sel       = [-40,-30,40,50] # Set to None to Do Point Analysis
#bbox_name      = "NAC"

#bbox_sel    =  [-40,-25,50,60] # Irminger
#bbox_name   = "IRM"

# bbox_sel   = [-70,-55,35,40] # Sargasso Sea
# bbox_name  = "SAR"


#%%

# Preprocess Files

ds_all = [ugeo_mod,ds_sst,ds_sss]
vunits = ["m/s"       ,"\degree C"  ,"psu"]
vnames = ["u_{geo}"   ,"SST"          ,"SSS"]


ds_all = [chk_dims(ds) for ds in ds_all] # Rename Dimensions

# Select Region or Point
if bbox_sel is None:
    print("Selecting a point!")
    locfn,loctitle = proc.make_locstring(lonf,latf)
    ds_pt       = [ds.sel(lon=lonf,lat=latf,method='nearest') for ds in ds_all]


else:
    print("Computing regional average!")
    locfn,loctitle = proc.make_locstring_bbox(bbox_sel)
    
    ds_pt       = [proc.sel_region_xr(ds,bbox_sel).mean('lat').mean('lon') for ds in ds_all]
    
    locfn       = "%s_%s"   % (bbox_name,locfn)
    loctitle    = "%s (%s)" % (bbox_name,loctitle)
    
#%% Move files into the array

#dsa_pt          = [preproc_ds(ds) for ds in ds_pt]
dsa_pt   = [ds - ds.mean('ens') for ds in ds_pt]
#arr_pt  = [ds.data.flatten() for ds in dsa_pt]

arr_pt          = [ds.data for ds in dsa_pt]
arr_pt_flatten  = [ds.flatten() for ds in arr_pt]

for ii in range(3):
    if np.any(np.isnan(dsa_pt[ii])):
        
        idens,idtime = np.where(np.isnan(dsa_pt[ii].data))
        idcomb       = np.where(np.isnan(arr_pt_flatten[ii]))[0][0]
        
        print("NaN Detected in arr %02i (ens=%02i, t=%s)" % (ii,idens+1,proc.noleap_tostr(dsa_pt[ii].time.isel(time=idtime))))
        arr_pt[ii][idens[0],idtime[0]] = 0
        arr_pt_flatten[ii][idcomb] = 0

#[print(np.any(np.isnan(ds))) for ds in arr_pt]
# replace the one in 
#arr_pt[2][32,219] = 0

#%% Compute the coherence


id_a = 0
id_b = 2



opt     = 1
nsmooth = 100
pct     = 0.10
CP,QP,freq,dof,r1_x,r1_y,PX,PY = yo.yo_cospec(arr_pt_flatten[id_a], #ugeo
                                              arr_pt_flatten[id_b], #sst
                                              opt,nsmooth,pct,
                                              debug=False,verbose=False,return_auto=True)



# Compute the confidence levels
CCX = yo.yo_speccl(freq,PX,dof,r1_x)
CCY = yo.yo_speccl(freq,PY,dof,r1_y)

# Compute Coherence
coherence_sq = CP**2 / (PX * PY)




#%% Plot the coherence

xtks        = np.arange(0,.12,0.02)
xtk_lbls    = [1/(x)  for x in xtks]
fig,axs     = plt.subplots(3,1,figsize=(12,10),constrained_layout=True)

for a in range(3):
    
    ax = axs[a]
    
    # Establish the plots
    if a == 0:
        plotvar = PX
        lbl     = vnames[id_a] #"$|u_{geo}|$"
        ylbl    = "$(%s)^2 \, cycles^{-1} \, mon^{-1}$" % vunits[id_a]
        plotCC  = CCX
    elif a == 1:
        plotvar = PY
        #lbl     = "SSS"
        #ylbl    = "$psu^2 \, cycles^{-1} \, mon^{-1}$"
        lbl     = vnames[id_b]
        ylbl    = "$(%s)^2 \, cycles^{-1} \, mon^{-1}$" % vunits[id_b]
        plotCC  = CCY
    elif a == 2:
        plotvar = coherence_sq
        lbl     = "Coherence"
        ylbl    = r"$%s \, %s \, cycles^{-1} \, mon^{-1}$" % (vunits[id_a],vunits[id_b])
    
    # Plot Frequency
    ax.plot(freq,plotvar,label="")
    if a < 2:
        ax.plot(freq,plotCC[:,0],label="",color="k",lw=0.75)
        ax.plot(freq,plotCC[:,1],label="",color="gray",lw=0.75,ls='dashed')
        
        
        
    
    
    
    if a == 2:
        ax.set_xlabel("cycles/mon")
    ax.set_title(lbl)
    #ax.set_xticks(xfreq)

    ax.axvline([1/12],color="lightgray",ls='dashed',label="Annual")
    ax.axvline([1/60],color="gray",ls='dashed',label="5-yr")
    ax.axvline([1/120],color="dimgray",ls='dashed',label="Decadal")
    ax.axvline([1/1200],color="k",ls='dashed',label="Cent.")
    if a == 0:
        ax.legend(fontsize=16,ncols=2)
    ax.set_ylabel(ylbl)
    ax.set_xticks(xtks)
    ax.set_xlim([0,0.1])
    ax.grid(True,ls='dotted')
    
    #ax2 = ax.twiny()
    #ax2.set_xticks(xtks,labels=xtk_lbls)
    #ax2.set_xlabel("")
    
    

#%% Do some standardization






