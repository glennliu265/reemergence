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

# Output is damping folder
datpath    = '/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/NATL_proc/'
input_path =  '/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/model_input/'
outpath    = input_path + 'damping/'
mpath      = input_path + "mld/"

vnames     = ["LHFLX","SHFLX"]
ncstr      = "CESM1LE_%s_NAtl_19200101_20050101_bilinear.nc"

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

Bscycle.isel(ensemble=0,mon=0).plot(vmin=-1,vmax=1),plt.show()


#%% Take monthly mean of fluxes then compute B
#ds_all  = [ds.rename({'month':'mon'}]
ds_savg = [ds.groupby('time.month').mean('time').rename({'month':'mon'}) for ds in ds_all]
Bscycle2 = ds_savg[1]/ds_savg[0]


savenames = ["%sCESM1LE_%s_NAtl_Hist_SAvg.nc" % (datpath,vnames[v]) for v in range(2)]
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

Bscycle = Bscycle.rename({'ensemble':"ens"})
Bscycle = Bscycle.rename("B")
Bscycle = Bscycle.transpose('mon','ens','lat','lon') # Tranpose to match other inputs

edict   = dict(B=dict(zlib=True))
savename = "%s/CESM1LE_BowenRatio_NAtl_19200101_20050101.nc" % outpath
if Bcorr:
    savename = proc.addstrtoext(savename,"_Bcorr3",adjust=-1)
Bscycle.to_netcdf(savename,encoding=edict)

# Save Ensemble Average
ensavg = Bscycle.mean('ens')
savename = "%s/CESM1LE_BowenRatio_NAtl_19200101_20050101_EnsAvg.nc" % outpath
if Bcorr:
    savename = proc.addstrtoext(savename,"_Bcorr3",adjust=-1)
ensavg.to_netcdf(savename,encoding=edict)