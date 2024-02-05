#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Using exponential fit to SST and SSS, prepare ocean and atmospheric damping terms

Note: This script runs on Astraeus, but output needs to be copied to stormtrack...
Should adapt so that the script can switch between the two

Uses output from [estimate_damping_fit]

Created on Sun Feb  4 14:28:47 2024

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

#%% Declare Paths and Variables

# Path to data and exponentially fitted parameters
datpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
ncname      = "CESM1_LENS_SST_SSS_lbd_exponential_fit_lagmax1to3.nc"

# Data Output Path
outpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/damping/"

# Figure Path
figpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/02_Figures/20240205/"
proc.makedir(figpath)

# Out Format Notes (to match the existing format created by...[prep_sm_inputs_SSS.py])
# CESM1_HTR_FULL_LHFLX_damping_nomasklag1_EnsAvg.nc [damping (mon x ens x lat x lon)]

# Indicate lags to include
lags   = [1,2,3]
lagstr = "lagsfit" + "".join([str(l) for l in lags])

# Convenience function
def save_ens_all_avg(ds,savename,edict,adjust=-1):
    """
    Parameters
    ----------
    ds       : Target Data Array with 'ens' dimension
    savename : STR Name to save to 
    edict    : TEncoding dictionary
    
    """
    
    # First, save for all members
    ds.to_netcdf(savename,encoding=edict)
    print("Saved variable to %s!" % savename)
    
    # Then, save for ens avg.
    ds_ensavg    = ds.mean('ens')
    sname_ensavg = proc.addstrtoext(savename,"_EnsAvg",adjust=adjust)
    ds_ensavg.to_netcdf(sname_ensavg,encoding=edict)
    print("Saved Ens Avg to %s!" % sname_ensavg)
    
    return None

def selpt_ds(ds,lonf,latf):
    return 
#ds.sel(lon=lonf,lat=latf,method='nearest')
#%% Load in the files

# Load 
ds      = xr.open_dataset(datpath+ncname).load() # [variable, lag, ens, mon, lat, lon]

# Select the lags that the exp function was fitted over, and take the mean estimate.
ds_lag  = ds.sel(lag_max=lags).mean('lag_max')

# Rearrange the Dimensions to match the format in model_input/damping
ds_lag  = ds_lag.lbd.transpose('vars','mon','ens','lat','lon')

# Separate and do some calculations
sss_lbd = ds_lag.sel(vars="SSS").rename('damping') *-1 # ['mon','ens','lat','lon']
sst_lbd = ds_lag.sel(vars="SST").rename('damping') *-1

# Make Encoding Dict
edict  = {"damping":{"zlib":True}}

#%% First save the overall damping fit to SST

savename = "%sCESM1_HTR_FULL_Expfit_SST_damping_%s.nc" % (outpath,lagstr)
save_ens_all_avg(sst_lbd,savename,edict)

#%% Next Save overall damping for SSS

savename = "%sCESM1_HTR_FULL_Expfit_SSS_damping_%s.nc" % (outpath,lagstr)
save_ens_all_avg(sss_lbd,savename,edict)


#%% Next, Estimate atmospheric damping (lbd_a) assuming lbd_sss = lbd_o

lbd_a_est = sst_lbd - sss_lbd
savename  = "%sCESM1_HTR_FULL_Expfit_lbda_damping_%s.nc" % (outpath,lagstr)
save_ens_all_avg(lbd_a_est*-1,savename,edict)
#proc.check_sum_ds([sst_lbd,-sss_lbd],sum_ds=lbd_a_est)

#%% Load in estimated lambda values

dt  = 3600*24*30
cp0 = 3996
rho = 1026

ds_stat   = "%sCESM1_HTR_FULL_qnet_damping_nomasklag1.nc" % outpath
lbda_stat = xr.open_dataset(ds_stat).damping

# Load other parameters
hmxl_htr  =  xr.open_dataset(outpath + "../mld/CESM1_HTR_FULL_HMXL_NAtl.nc").h

# Convert to damping timescale...
lbda_stat_conv = lbda_stat / (hmxl_htr*cp0*rho) * dt

# Compute Entrainment damping
hclim = hmxl_htr.values # (12, 42, 96, 89)

nens = hclim.shape[1]
beta_all=[]
for e in range(nens):
    hin = hclim[:,e,:,:].transpose(2,1,0)
    beta = scm.calc_beta(hin)
    beta_all.append(beta)
beta_all = np.array(beta_all) # [lon x lat x mon]




#%% Debug values

lonf,latf=[-30,50]
ds_in    =[sst_lbd,sss_lbd,lbd_a_est,lbda_stat_conv]
labs     = ["SST Exp. fit","SSS Exp. fit ($\lambda^o$)","$\lambda^a$ (Exp Fit.)","$\lambda^a$ (Cov Est.)"]
cols     = ["r","cornflowerblue","orange","k"]
dspt = [proc.selpt_ds(ds,lonf,latf) for ds in ds_in] # [Mon x Ens]

locfn,loctitle=proc.make_locstring(lonf,latf)

hlon = hmxl_htr.lon.values
hlat = hmxl_htr.lat.values
khlon,khlat = proc.find_latlon(lonf,latf,hlon,hlat)

_,nens = dspt[0].shape


#%%

mons3=proc.get_monstr()
fig,ax= viz.init_monplot(1,1,)

ensavgs = []
for ii in range(4):
    #ax = axs[ii]
    for e in range(nens):
        ax.plot(mons3,dspt[ii].isel(ens=e),label="",alpha=.05,color=cols[ii],zorder=1)
    ax.plot(mons3,dspt[ii].mean('ens'),label=labs[ii],color=cols[ii])
    
    ensavgs.append(dspt[ii].mean('ens'))
    

for e in range(nens):
    ax.plot(mons3,beta_all[e,khlon,khlat,:,],label="",color='darkblue',zorder=1,alpha=0.05)

ax.plot(mons3,beta_all[:,khlon,khlat,:,].mean(0),label="Entrainment ($w_e/h$)",color='darkblue',zorder=1,alpha=1)


ax.legend(ncol=3)
    



ax.set_title("Damping @ %s" % loctitle)

savename = "%sDamping_Estimates_AllEns_SPGPoint.png" % (figpath) 
plt.savefig(savename,dpi=150,bbox_inches='tight')




#%% Load and check with ACFs
datpath_ac  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
varnames    = ("SST","SSS")

ds_all = []
#ac_all = []
for v in range(2):
    ds = xr.open_dataset("%sHTR-FULL_%s_autocorrelation_thres0.nc" % (datpath_ac,varnames[v]))
    
    ds  = ds.sel(thres="ALL").sel(lon=lonf,lat=latf,method='nearest').load()# [ens lag mon]
    ds_all.append(ds)
    #ac_all.append(ds[varnames[v]].values) 
    

#%% Check Exponential Fit, comparing with CESM
xtks   = np.arange(0,37,3)

expf3      = lambda t,b: np.exp(b*t)         # No c and A

lags = np.arange(37)

kmonth = 7

for kmonth in range(12):
    fig,ax = plt.subplots(1,1)
    
    ax,_ = viz.init_acplot(kmonth,xtks,lags,ax=ax)
    
    
    [ax.plot(lags,expf3(-ensavgs[dd].isel(mon=kmonth).values,lags),label=labs[dd],c=cols[dd]) for dd in range(3)]
    
    ax.plot(lags,ds_all[0].SST.mean('ens').isel(mon=kmonth),label="SST (CESM Ens Avg)",color='k',ls='dashed')
    ax.plot(lags,ds_all[1].SSS.mean('ens').isel(mon=kmonth),label="SSS (CESM Ens Avg)",color='gray',ls='dashed')
    
    
    
    #[ax.plot(mons3,ensavgs[dd].values,label=labs[dd],c=cols[dd]) for dd in range(3)]
    
    ax.legend()
    
    savename = "%sDamping_Estimates_Ensavg_SPGPoint_mon%02i.png" % (figpath,kmonth+1) 
    plt.savefig(savename,dpi=150,bbox_inches='tight')

#%%

sss_lbd = ds.lbd.sel(vars='SSS').transpose('lag_max','mon','ens','lat','lon')#.mean()#.values #  [lag, ens, mon, lat, lon]]
sst_lbd = ds.lbd.sel(vars='SST').transpose('lag_max','mon','ens','lat','lon')#.values

# Select the lags that the exp function was fitted over, and take the mean estimate.
sss_lags = sss_lbd.sel(lag_max=lags).mean('lag_max')
sst_lags = sss_lbd.sel(lag_max=lags).mean('lag_max')



