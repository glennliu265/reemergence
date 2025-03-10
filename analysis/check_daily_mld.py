#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Check Daily Mixed-Layer Depth Variability

Created on Thu Sep 12 16:17:53 2024

@author: gliu

"""

import numpy as np
import xarray as xr
import sys
import time
import matplotlib.pyplot as plt

# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "stormtrack"

# First Load the Parameter File
sys.path.append("../")
import reemergence_params as rparams

# Paths and Load Modules
pathdict   = rparams.machine_paths[machine]

sys.path.append(pathdict['amvpath'])
sys.path.append(pathdict['scmpath'])
from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx

# Set needed paths
figpath     = pathdict['figpath']
proc.makedir(figpath)
input_path  = pathdict['input_path']
output_path = pathdict['output_path']
procpath    = pathdict['procpath']


#%% Indicate Paths

#figpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20240411/"
datpath   = pathdict['raw_path']
outpath   = pathdict['input_path']+"forcing/"
rawpath   = pathdict['raw_path']

#%% Load Point Information

# Get Point Info
pointset    = "PaperDraft02"
ptdict      = rparams.point_sets[pointset]
ptcoords    = ptdict['bboxes']
ptnames     = ptdict['regions']
ptnames_long = ptdict['regions_long']
ptcols      = ptdict['rcols']
ptsty       = ptdict['rsty']


#%%
dpathdaily      = "/stormtrack/data4/glliu/01_Data/CESM1_LE/HMXL/daily/"
ncdaily         = "b.e11.B20TRC5CNBDRD.f09_g16.002.pop.h.nday1.HMXL_2.19200102-20051231.nc"

ds_daily = xr.open_dataset(dpathdaily + ncdaily)

# ==================
#%% Select a point
# ==================

lonf = 330
latf = 50
st   = time.time()
hdaily_pt = proc.find_tlatlon(ds_daily,lonf,latf)
hdaily_pt = hdaily_pt.HMXL_2.load()
print("Loaded daily file in %.2fs" % (time.time()-st))

# Need to crop time again (because I think I croppped it incorrectly)
hdaily_crop = hdaily_pt.sel(time=slice("1921-01-01","2005-12-31"))

#%% Group by month
# Compute max, min (range)
# Compute standard deviation (intramonthly)

hd    = hdaily_pt.data # Year x mon x day
ntime = hd.shape[0]


nyr   = int(ntime/(12*30))

hd_reshape = hd.reshape(nyr,12,30)

#%% Do a quick guess

# Compute standard deviation over all days of all years in a given month
hd_monstd   = hdaily_pt.groupby('time.month').std('time')
hd_monmax   = hdaily_pt.groupby('time.month').max('time')
hd_monmin   = hdaily_pt.groupby('time.month').min('time')
hd_monmean  = hdaily_pt.groupby('time.month').mean('time')

mons3 = proc.get_monstr()


opath       =  "/stormtrack/data4/glliu/01_Data/CESM1_LE/HMXL/daily/"
savename    = "%sDaily_HMXL_lon%03i,lat%03i.nc" % (opath,lonf,latf)
hdaily_pt.to_netcdf(savename)



#%%

fig,ax = viz.init_monplot(1,1)

ax.plot(mons3,hd_monmean,label="Mean",color="k",lw=1.5)
ax.fill_between(mons3,hd_monmean-hd_monstd,hd_monmean+hd_monstd,alpha=0.15)


ax.scatter(mons3,hd_monmax,c='r',marker="x",label="Max",lw=1.5)
ax.scatter(mons3,hd_monmin,c='b',marker="o",label="Min",lw=1.5)
ax.legend()
plt.show()


#%% ok now do the same for all points but crop to a region 

bbox    = [-80,0,20,65]
st = time.time()
ds_reg = proc.sel_region_xr_cv(ds_daily,bbox)
ds_reg = ds_reg.load()
print("Data Loaded in %.2fs" % (time.time()-st))

#%% Now do the same computations above for max, min, and intramonthly standard deviation

hd_monmean  = ds_reg.groupby('time.month').mean('time').rename("monmean")
hd_monstd   = ds_reg.groupby('time.month').std('time').rename("monstd")
hd_monmax   = ds_reg.groupby('time.month').max('time').rename("monmax")
hd_monmin   = ds_reg.groupby('time.month').min('time').rename("monmin")

hd_sumstat  = xr.merge([hd_monmean,hd_monstd,hd_monmax,hd_monmin])
edict       = proc.make_encoding_dict(hd_sumstat)

savename    = "%sCESM1_HMXL_Daily_SumStat_NAtl.nc" % dpathdaily


#%% Loop for a few locations


npts = 3

for pp in range(npts):
    
    lonf,latf = ptcoords[pp][0],ptcoords[pp][1]
    if lonf < 0:
        lonf = lonf + 360
    
    
    st   = time.time()
    hdaily_pt = proc.find_tlatlon(ds_daily,lonf,latf)
    hdaily_pt = hdaily_pt.HMXL_2.load()
    print("Loaded daily file in %.2fs" % (time.time()-st))
    
    
    # Compute standard deviation over all days of all years in a given month
    hd_monstd = hdaily_pt.groupby('time.month').std('time')
    
    hd_monmax = hdaily_pt.groupby('time.month').max('time')
    hd_monmin = hdaily_pt.groupby('time.month').min('time')
    hd_monmean = hdaily_pt.groupby('time.month').mean('time')
    
    mons3 = proc.get_monstr()
    
    
    opath    =  "/stormtrack/data4/glliu/01_Data/CESM1_LE/HMXL/daily/"
    savename = "%sDaily_HMXL_lon%03i,lat%03i.nc" % (opath,lonf,latf)
    hdaily_pt.to_netcdf(savename)

#
# %% Try Selecting a Region
#
tlon = ds_daily.TLONG#.data
tlat = ds_daily.TLAT#.data

bbox         = [-80,0,0,65]
ds_daily_reg = proc.sel_region_cv(tlon,tlat,ds_daily.HMXL_2.transpose('nlat','nlon','time'),bbox)



#%% Do a pointwise loop

bbox         = [-80,0,0,65]
ds_lonr      = proc.sel_region_xr(xr.open_dataset(outpath + "CESM1_HTR_FULL_qnet_NAtl_EnsAvg.nc"),bbox)
lonr = ds_lonr.lon.load()
latr = ds_lonr.lat.load()

for o in range(len(lonr)):
    for a in range(len(latr)):
        lonf = lonr[o]
        latf = latr[a]
        if lonf < 0:
            lonf = lonf + 360
        
        
        st   = time.time()
        hdaily_pt = proc.find_tlatlon(ds_daily,lonf,latf)
        hdaily_pt = hdaily_pt.HMXL_2.load()
        print("Loaded daily file in %.2fs" % (time.time()-st))
        
        
        # Compute standard deviation over all days of all years in a given month
        hd_monstd  = hdaily_pt.groupby('time.month').std('time')
        hd_monmax  = hdaily_pt.groupby('time.month').max('time')
        hd_monmin  = hdaily_pt.groupby('time.month').min('time')
        hd_monmean = hdaily_pt.groupby('time.month').mean('time')
        
        mons3 = proc.get_monstr()
        
        
        opath    =  "/stormtrack/data4/glliu/01_Data/CESM1_LE/HMXL/daily/"
        savename = "%sDaily_HMXL_lon%03i,lat%03i.nc" % (opath,lonf,latf)
        hdaily_pt.to_netcdf(savename)
    
#%% Next Section, combine all the points

from tqdm import tqdm
nlon = len(lonr)
nlat = len(latr)
basinwide_h = np.zeros((4,12,nlat,nlon)) * np.nan

for o in tqdm(range(len(lonr))):
    for a in range(len(latr)):
        lonf = lonr[o]
        latf = latr[a]
        if lonf < 0:
            lonf = lonf + 360
            
        opath    =  "/stormtrack/data4/glliu/01_Data/CESM1_LE/HMXL/daily/"
        savename = "%sDaily_HMXL_lon%03i,lat%03i.nc" % (opath,lonf,latf)
        
        ds                 = xr.open_dataset(savename).HMXL_2.load()
        
        basinwide_h[0,:,a,o] = ds.groupby('time.month').mean('time').data
        basinwide_h[1,:,a,o] = ds.groupby('time.month').std('time').data
        basinwide_h[2,:,a,o] = ds.groupby('time.month').min('time').data
        basinwide_h[3,:,a,o] = ds.groupby('time.month').max('time').data

#%% Next, place into Data Array and saven

coords = dict(
    metric= ["mean","std","min","max"],
    mon   = np.arange(1,13,1),
    lat  = latr,
    lon  = lonr,
    )

#coords_xy = dict(nlat  = np.arange(nlat),nlon  = np.arange(nlon))

da_out    = xr.DataArray(basinwide_h,coords=coords,dims=coords,name="stats")

#da_tlon   = xr.DataArray(lonr)


savename = opath + "CESM1_Daily_HMXL_Ens02_SumStats.nc"
edict    = proc.make_encoding_dict(da_out)

da_out.to_netcdf(savename,encoding=edict)

        
        
        
        

