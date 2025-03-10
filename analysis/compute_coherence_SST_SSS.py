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


#%% Load Ice Mask ETC

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


id_a = 1
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




def pointwise_coherence(ts1,ts2,opt=1,nsmooth=100,pct=0.10):
    
    if np.any(np.isnan(ts1)) or np.any(np.isnan(ts2)):
        return np.nan
        
    else:
        
        CP,QP,freq,dof,r1_x,r1_y,PX,PY = yo.yo_cospec(ts1, #ugeo
                                                      ts2, #sst
                                                      opt,nsmooth,pct,
                                                      debug=False,verbose=False,return_auto=True)
        
        # Compute Coherence
        coherence_sq = CP**2 / (PX * PY)
        
        return coherence_sq
    
    
coherence_sq = pointwise_coherence(arr_pt_flatten[id_a],arr_pt_flatten[id_b])



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


#%% Preprocess

dsvars = ds_all[1:] # Just Take SST and SSS

# Anomalize
dsanom = [proc.xrdeseason(ds) for ds in dsvars]

# detrend
dsanom = [ds - ds.mean('ens') for ds in dsanom]


#%% Try computing coherence for SST and SSS overall
nsmooth  = 10
opt      = 1
pct      = 0.10

calc_coh = lambda ts1,ts2: pointwise_coherence(ts1,ts2,nsmooth=nsmooth,opt=opt,pct=pct)

# Compute Spectra
st = time.time()
coh_ens = xr.apply_ufunc(
    calc_coh,  # Pass the function
    dsanom[0],  # The inputs in order that is expected
    dsanom[1],
    # Which dimensions to operate over for each argument...
    input_core_dims=[['time'],['time']],
    output_core_dims=[['freq'],],  # Output Dimension
    exclude_dims=set(("freq",)),
    vectorize=True,  # True to loop over non-core dims
)
print("Completed calculations in %.2fs" % (time.time()-st))




# # Need to Reassign Freq as this dimension is not recorded
dt              = 3600*24*30
ts1             = dsanom[0].isel(ens=0).isel(lon=22,lat=22).values

sps             = yo.yo_spec(ts1, opt, nsmooth, pct, debug=False)
freq_ts         = sps[1]/dt
coh_ens['freq'] = freq_ts



#%% Save the file



savename = "%sCESM1_NATL_SST_SSS_Coherence_nsmooth%03i.nc" % (rawpath,nsmooth)
coh_ens.to_netcdf(savename)

#%% Plot Coherence at a sele ted point
coh_ens_pt = coh_ens.sel(lon=lonf,lat=latf,method='nearest')



fig,ax  = plt.subplots(1,1)
plotvar = coh_ens_pt
ax.plot(plotvar.freq*dt,plotvar.mean('ens'),label="Ens Mean")

for ii in range(42):
    ax.plot(plotvar.freq*dt,plotvar.isel(ens=ii),label="",alpha=0.05)

ax.plot(freq,coherence_sq,label="All Ens",lw=.55)
ax.legend()

ax.set_xlim([0,0.1])
ax.axvline([1/(86*12)])

# Checking results from below
ax.axvline([7.477e-10*dt],c='r',label="50yr")


#%% Make map of coherence at 50 years

coh_lf = coh_ens.sel(freq=1/(1*12*dt),method='nearest')

cints  = np.arange(0,1.1,0.1)
# Initialize Plot and Map

fig,ax,_    = viz.init_orthomap(1,1,bboxplot,figsize=(24,10))
ax          = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="lightgray",fontsize=fsz_tick)

plotvar     = coh_lf.mean('ens')
pcm         = ax.contourf(plotvar.lon,plotvar.lat,plotvar,transform=proj,levels=cints,cmap='cmo.tempo')
cl          = ax.contour(plotvar.lon,plotvar.lat,plotvar,transform=proj,levels=cints,colors="k",
                         linewidths=0.75)
ax.clabel(cl,fontsize=fsz_tick)


cb          = viz.hcbar(pcm,ax=ax)
cb.ax.tick_params(labelsize=fsz_tick)
cb.set_label("Coherence Squared",fontsize=fsz_axis)

ax.set_title("SST-SSS Coherence Squared \n@ Period = %.f years" % (1/(coh_lf.freq*dt*12).item()),fontsize=fsz_axis)


# Plot Gulf Stream Position
ax.plot(ds_gs2.lon.mean('mon'),ds_gs2.lat.mean('mon'),transform=proj,lw=2.5,c='k',ls='dashdot')

# Plot Ice Edge
ax.contour(icemask.lon,icemask.lat,mask_plot,colors="cyan",linewidths=2.5,
           transform=proj,levels=[0,1],)
