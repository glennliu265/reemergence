#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calc_Bowen_Ratio


Created on Mon Apr 15 18:09:39 2024

@author: gliu
"""
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import sys


#%%
# stormtrack
amvpath = "/home/glliu/00_Scripts/01_Projects/00_Commons/" # amv module
scmpath = "/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/" # scm module

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl


#%%


dataset_name = "cesm1_htr_5degbilinear"
# Output is damping folder
if dataset_name == "CESM1_HTR":
    datpath    = '/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/NATL_proc/'
    input_path =  '/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/'
    outpath    = input_path + 'damping/'
    mpath      = input_path + "mld/"

    
    ncstr           = "CESM1LE_%s_NAtl_19200101_20050101_bilinear.nc"
    outname_nc      = datpath + "CESM1LE_%s_NAtl_Hist_SAvg.nc"
    savename_bowen  = outpath + "CESM1LE_BowenRatio_NAtl_19200101_20050101.nc"
elif dataset_name == "cesm1_htr_5degbilinear":
    
    datpath    = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/output/proc/"
    input_path =  '/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/'
    outpath    = input_path + 'damping/'
    mpath      = input_path + "mld/"
    ncstr      = "cesm1_htr_5degbilinear_%s_Global_1920to2005.nc"
    
    bbox_crop  = [-90,0,0,90]
    regcrop    = "NAtl"
    
    outname_nc      = datpath + dataset_name + "_%s_" + "Global_Hist_SAvg.nc"
    savename_bowen  = outpath + dataset_name + "_BowenRatio_NAtl_19200101_20050101.nc"
    
vnames     = ["LHFLX","SHFLX"]
Bcorr      = True
#%% Read in LHFLX and SHFLX

ds_all = []
for vn in vnames:
    ds = xr.open_dataset(datpath+ncstr % vn)[vn].load()
    ds_all.append(ds)
    
#%% Calculate B first then take monthly mean

# Time varying B
B = ds_all[1] / ds_all[0]

# Look at mean
Bscycle = B.groupby('time.month').mean('time').rename({'month':'mon'})


#Bscycle.isel(ens=0,mon=0).plot(vmin=-1,vmax=1),plt.show()


#%% Take monthly mean of fluxes then compute B
#ds_all  = [ds.rename({'month':'mon'}]
ds_savg = [ds.groupby('time.month').mean('time').rename({'month':'mon'}) for ds in ds_all]
Bscycle2 = ds_savg[1]/ds_savg[0]


savenames = [outname_nc % (vnames[v]) for v in range(2)]
[ds_savg[s].to_netcdf(savenames[s]) for s in range(2)]
#%% Visualize/Check diff between methods (seems minimal)

diff = (Bscycle-Bscycle2).isel(ensemble=0,mon=1)
plt.pcolormesh(diff,vmin=-.2,vmax=.2,cmap="RdBu_r"),plt.colorbar(),plt.show()

#%% Perform a correction for large Bowen Ratios
if Bcorr:
    def interp_scycle(ts,thres=3,):
        
        ts       = np.abs(ts)
        idexceed = np.where(ts > thres)[0]
        for ii in idexceed:
            if ii + 1 > 11:
                iip1 = 0
            else:
                iip1 = 11
            ts[ii] = np.interp(1,[0,2],[ts[ii-1],ts[iip1]])
        return ts
    
    Bscycle = xr.apply_ufunc(
        interp_scycle,  # Pass the function
        Bscycle,  # The inputs in order that is expected
        # Which dimensions to operate over for each argument...
        input_core_dims =[['mon'],],
        output_core_dims=[['mon'],],  # Output Dimension
        #exclude_dims=set(("",)),
        vectorize=True,  # True to loop over non-core dims
    )
    

#%% Save Output
if 'ensemble' in list(Bscycle.dims):
    Bscycle = Bscycle.rename({'ensemble':"ens"})
Bscycle = Bscycle.rename("B")
Bscycle = Bscycle.transpose('mon','ens','lat','lon') # Tranpose to match other inputs

# Flip longitude and crop to region
Bscycle180  = proc.lon360to180_xr(Bscycle)
Bscycle_reg = proc.sel_region_xr(Bscycle180,bbox_crop)


edict   = dict(B=dict(zlib=True))
savename = savename_bowen
if Bcorr:
    savename = proc.addstrtoext(savename,"_Bcorr3",adjust=-1)
Bscycle_reg.to_netcdf(savename,encoding=edict)

# Save Ensemble Average
ensavg = Bscycle.mean('ens')
savename = proc.addstrtoext(savename,"_EnsAvg",adjust=-1)
if Bcorr:
    savename = proc.addstrtoext(savename,"_Bcorr3",adjust=-1)
ensavg.to_netcdf(savename,encoding=edict)