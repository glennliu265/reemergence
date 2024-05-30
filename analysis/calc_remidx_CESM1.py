#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute Re-emergence Index (copied from visualize_reemergence_CMIP6.py)
Created on Fri May 24 11:07:31 2024

@author: gliu

"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
from tqdm import tqdm
import copy
import glob
import matplotlib as mpl

#%% Import Custom Modules

amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx

#%%

datpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CMIP6/proc/"
figpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/02_Figures/20240531/"
proc.makedir(figpath)

#%%



#%% Try for CESM1

cesm1ncs  = ['HTR-FULL_SST_autocorrelation_thres0.nc','HTR-FULL_SSS_autocorrelation_thres0.nc',""]
cesm1path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
vnames    = ["SST","SSS"]

ds_opn    = [xr.open_dataset(cesm1path+nc) for nc in cesm1ncs] # [thres x ens x lag x mon x lat x lon]
ds_opn    = [ds.isel(thres=2) for ds in ds_opn]


dsadd     = xr.open_dataset(cesm1path + "CESM1_1920to2005_TEMPACF_lag00to60_ALL_ensALL.nc").load()
ds        = dsadd


ds = ds.rename({'lags':'lag','mons':'mon'})
#ds_opn = ds_opn + [dsadd,]

#%%
# Load out some variables
#ds  = ds_opn[0]

ds  = ds.transpose('ens','lag','mon','lat','lon')
lon = ds.lon.values
lat = ds.lat.values
lag = ds.lag.values # might be lag instead of lags, whoops
ens = ds.ens.values
mon = ds.mon.values 


#%% Additional Plotting settings
bboxplot  = [-80,0,0,65]
mpl.rcParams['font.family'] = 'Avenir'
mons3     = proc.get_monstr(nletters=3)

fsz_tick  = 18
fsz_axis  = 20
fsz_title = 24

#%%

rhocrit = proc.ttest_rho(0.05,1,86)

vv          = 0
vname       = vnames[vv]
acf         = #ds_opn[vv][vname].values # [Ens Lag Mon Lat Lon]
save_remidx = True

norm_rem    = False
sigmask     = False

plotbbox = True



if vv == 0:
    cmapin='cmo.dense'
else:
    cmapin='cmo.deep'

nens,_,_,nlat,nlon=acf.shape
remidxs = np.zeros([12,3,nens,nlat,nlon]) * np.nan
for kmonth in range(12):
    print(kmonth)
    
    mmcorr = proc.calc_remidx_simple(acf,kmonth,monthdim=2,lagdim=1) # [min/max,yr,mon,lat,lon]
    
    if mmcorr.shape[1] > 3:
        mmcorr = mmcorr[:,:3,:,:,:]
    
    if sigmask:
        rempeaks  = mmcorr[1,:,:,:,:].mean(1) # [year x lat x lon]
        sigmaskin = rempeaks > rhocrit
        
        
    remidx = mmcorr[1,...] - mmcorr[0,...] #  (3, 42, 69, 65)
    if norm_rem:
        remidx = remidx/mmcorr[1,...]
    #remidxs.append(remidx)
    remidxs[kmonth,...] = remidx.copy()
    
    
    
    #%Plot 3-year re-emergence, 1 ensemble Avg
    #e = 3
    bbplot2 = [-80,0,15,65]
    levels  = np.arange(0,0.55,0.05)
    plevels = np.arange(0,0.6,0.1)
    
    fig,axs,mdict = viz.init_orthomap(1,3,bbplot2,figsize=(16,8),constrained_layout=True,centlat=45)
    
    for yy in range(3):
        
        ax = axs[yy]
        blb = viz.init_blabels()
        if yy !=0:
            blb['left']=False
        else:
            blb['left']=True
        blb['lower']=True
        ax           = viz.add_coast_grid(ax,bboxplot,fill_color="lightgray",fontsize=20,blabels=blb,
                                        fix_lon=np.arange(-80,10,10),fix_lat=np.arange(0,70,10),grid_color="k")
        
        plotvar = remidx[yy,:,:,:].mean(0)
        
        if sigmask:
            smsk    = sigmaskin[yy,:,:]
        
        pcm = ax.contourf(lon,lat,plotvar,cmap=cmapin,levels=levels,transform=mdict['noProj'],extend='both')
        cl = ax.contour(lon,lat,plotvar,colors='darkslategray',linewidths=.5,linestyles='solid',levels=levels,transform=mdict['noProj'])
        
        if sigmask:
            mskk = viz.plot_mask(lon,lat,smsk.T,ax=ax,proj=mdict['noProj'],geoaxes=True,
                                 markersize=.75)
        
        #ax.clabel(cl,plevels,fontsize=fsz_tick-8)
        ax.set_title("Year %i" % (yy+1),fontsize=fsz_title)
        
        if plotbbox:
            if vv == 0:
                bbsel = [-45,-25,50,60]
                bcol  = "orange"
            elif vv == 1:
                bbsel = [-70,-55,30,40]
                bcol  = "midnightblue"
            if yy == 0:
                zz = viz.plot_box(bbsel,ax=ax,color=bcol,linewidth=2.5)
        
    
    cb = fig.colorbar(pcm,ax=axs.flatten(),fraction=0.0105,pad=0.01)
    cb.ax.tick_params(labelsize=fsz_tick)
    cb.set_label("%s Re-emergence Index" % vname,fontsize=fsz_axis)
    
    savename = "%sCESM1_%s_RemIdx_normrem%0i_mon%02i_EnsAvg.png" % (figpath,vname,norm_rem,kmonth+1)
    plt.savefig(savename,dpi=200,bbox_inches='tight',transparent=True)

if save_remidx:
    coords   = dict(mon=np.arange(1,13,1),yr=np.arange(1,4,1),
                    ens=np.arange(1,43,1),lat=lat,lon=lon)
    da_rem   = xr.DataArray(remidxs,coords=coords,dims=coords,name="rei")
    edict    = {'rei':{'zlib':True}}
    outpath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/sm_experiments/%s_CESM/Metrics/REI_Pointwise.nc" % (vname)
    da_rem.to_netcdf(outpath,encoding=edict)
    
    
