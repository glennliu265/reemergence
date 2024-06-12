#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
------------------------------------
Compare SST SSS coupling experiments
------------------------------------
copied from viz_SST_SSS_coupling

(1) Load in SST and SSS timeseries from each experiment
(2) Load in SST and SSS from CESM1
(3) Compute the ACFs (monthly)
(4) Examine the following:
    a. Autocorrelation
    b. Cross-correlation
    c. High pass/Low pass cross correlation

~~~ ~~~ ~~~ ~~~ 
Created on Mon Jun 10 14:14:38 2024

@author: gliu
"""


import xarray as xr
import numpy as np
import matplotlib as mpl

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

"""
SST-SSS, Effect of White Noise and Coupling
(1) 

"""

# Paths and Experiment
expnames     = ['SST_SSS_LHFLX','SST_SSS_LHFLX_1','SST_SSS_LHFLX_2','SST_SSS_LHFLX_2_neg',
                'SST_SSS_LHFLX_DiffWn','SST_SSS_LHFLX_DiffT',"SST_SSS_LHFLX_DiffAll",'SST_SSS_LHFLX_2_noLbdE']


expcols = ["k","gray","goldenrod",'orange',"blue","salmon","magenta",'cyan']
expls   = ["solid",'dashed','dotted','dashed','dotted',"dashed","dotted",'solid']


# metrics_path = output_path + expname + "/Metrics/" 
# exp_output   = output_path + expname + "/Output/" 

vnames       = ["SST","SSS"]
proc.makedir(figpath)

#%% Define functions for loading/convenience

def load_exp_cpl(expname,output_path):
    # Loads coupled experiment [expname] to a list with [SST,SSS], where SST
    # and SSS is [run x time]
    #metrics_path = output_path + expname + "/Metrics/" 
    exp_output   = output_path + expname + "/Output/" 
    vnames       = ["SST","SSS"]
    
    ds_all  = []
    var_all = []
    for vv in range(2):
        
        globstr = "%s%s_runid*.nc" % (exp_output,vnames[vv])
        nclist  = glob.glob(globstr)
        nclist.sort()
        ds      = xr.open_mfdataset(nclist,combine='nested',concat_dim="run").load()
        
        if len(ds.lat) > 1: # Related to SSS error where 2 lat values are saved...
            print("Two latitude dimensions are detected... fixing.")    
            remake_ds = []
            for nr in range(len(ds.run)):
                invar = ds.isel(run=nr)[vnames[vv]]
                
                if np.all(np.isnan(invar.isel(lat=0))): 
                    klat = 1
                if np.all(np.isnan(invar.isel(lat=1))):
                    klat = 0
                print("Non-NaN Latitude Index was %i for run %i" % (klat,nr))
                invar = invar.isel(lat=klat)
                #invar['lat'] = 50.
                remake_ds.append(invar.values.copy())
            coords = dict(run=ds.run,time=ds.time)
            ds     = xr.DataArray(np.array(remake_ds).squeeze(),coords=coords,name=vnames[vv])
        else:
            ds = ds[vnames[vv]]
        
        #.sel(lat=50.42,method='nearest')
        ds_all.append(ds)
        var_all.append(ds.values.squeeze()) # [Run x Time]
    
    return ds_all,var_all,


# def leadlag_corr(varbase,varlag,lags,corr_only=False):
#     ntime = varbase.shape[0]
#     nlags = len(lags)
#     # Lags
#     leadcorrs = []
#     lagcorrs  = []
#     for ii in range(nlags):
#         lag     = lags[ii]
#         lagcorr  = np.corrcoef(varbase[:(ntime-lag)],varlag[lag:])[0,1]
#         leadcorr = np.corrcoef(varbase[lag:],varlag[:(ntime-lag)])[0,1]
#         lagcorrs.append(lagcorr)
#         leadcorrs.append(leadcorr)
#     leadlagcorr = np.concatenate([np.flip(leadcorrs)[:-1],lagcorrs])
#     leadlags    = np.concatenate([np.flip(-1*lags)[:-1],lags],)
#     if corr_only:
#         return leadlagcorr
#     return leadlags,leadlagcorr

#%% Load the variables
nexps     = len(expnames)
var_byexp = [] # [exp][sst/sss][run x time]
for ex in range(nexps):
    print(expnames[ex])
    _,vall = load_exp_cpl(expnames[ex],output_path)
    var_byexp.append(vall)
    
    
nruns,ntime = var_byexp[0][0].shape
    
    
#%% Compute the ACFs for each experiment, variable, and run
lags     = np.arange(37)
nlags    = len(lags)
acfs_all = np.zeros((nexps,2,nruns,12,nlags)) # [exp x variable x run x base month x lag]

for ex in range(nexps):
    print(expnames[ex])
    for vv in range(2):
        print("\t%s" % (vnames[vv]))
        for rr in tqdm.tqdm(range(10)):
            
            var_in   = var_byexp[ex][vv][rr,:]
            acfs_mon = scm.calc_autocorr_mon(var_in,lags,verbose=False,return_da=False)
            
            acfs_all[ex,vv,rr,:,:] = acfs_mon.copy()
            
#%% Plot the ACFs for a selected variable/basemonth

kmonth  = 1
xtks    = np.arange(37)

plotboth = False

if plotboth:
    fig,axs = plt.subplots(2,1,constrained_layout=True,figsize=(12,8))

    for vv in range(2):
        ax = axs[vv]
        ax,_ = viz.init_acplot(kmonth,xtks,lags,ax=ax,title="")
        
        for ex in range(nexps):
            
            plotacf = acfs_all[ex,vv,:,kmonth,:]    # 
            plotacf = plotacf.mean(0)               # Take mean along run dimension [lags,]
            
            ax.plot(lags,plotacf,label=expnames[ex],c=expcols[ex],ls=expls[ex],lw=2.5)
        if vv == 0:
            ax.legend()
else:
    fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12,4.5))
    
    ax,_ = viz.init_acplot(kmonth,xtks,lags,ax=ax,title="")
    
    for ex in range(nexps):
        
        plotacf = acfs_all[ex,vv,:,kmonth,:]    # 
        plotacf = plotacf.mean(0)               # Take mean along run dimension [lags,]
        
        ax.plot(lags,plotacf,label=expnames[ex],c=expcols[ex],ls=expls[ex],lw=2.5)
    ax.legend(ncol=2)
    
#%% Check the cross correlation/relationship between variables

#hpf       = None # Set to None to not do a high-pass filter
hicutoff  = 6 # Number of months
hipass    = lambda x: proc.lp_butter(x,hicutoff,6,btype='highpass')


ccfs_all = np.zeros((nexps,nruns,nlags*2-1)) # [exp x variable x run x base month x lag]

ccfs_all_hp = ccfs_all.copy() # high pass output
for ex in range(nexps):
    
    print(expnames[ex])
    for rr in range(nruns):
        sst_in        = var_byexp[ex][0][rr,:]
        sss_in        = var_byexp[ex][1][rr,:]
        
        sst_hp,sss_hp = [hipass(ts) for ts in [sst_in,sss_in]]
        if ex == 0:
            leadlags,ccf         = proc.leadlag_corr(sst_in,sss_in,lags,corr_only=False)
        else:
            ccf = proc.leadlag_corr(sst_in,sss_in,lags,corr_only=True)
        ccfs_hp = proc.leadlag_corr(sst_hp,sss_hp,lags,corr_only=True)
        ccfs_all[ex,rr,:] = ccf.copy()
        ccfs_all_hp[ex,rr,:] = ccfs_hp.copy()
            
#%% Plot the lead lag correlation (Copied fully from viz_SST_SSS_coupling)


xtks      = np.arange(-36,37,3)
add_alpha = False
plothp    = False
#dcols     = ["k","orange"]
#dnames    = ["CESM1","Stochastic Model"]
zoom      = False

ls_filter = ["solid",'dashed']

# cmap      = mpl.colormaps['tab10']
# threscols = cmap(np.linspace(0,1,nthres))
# add_alpha = False

fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12,4))


# for ss in range(2):
#     for zz in range(2):
#         if zz == 0:
#             inset = crosscorrs_byset
#             lab   = "Raw"
#         elif zz == 1:
#             inset = crosscorrs_byset_hipass
#             lab   = "High Pass"
        
#         threscorr = inset[ss]
    
#         mu        = threscorr.mean('ens')
        
#         ax.plot(leadlags,mu,color=dcols[ss],ls=ls_filter[zz],
#                 label="%s (%s)" % (dnames[ss],lab),zorder=3,lw=2.5)   
#         if add_alpha:
#             sigma = threscorr.std('ens')
#             ax.fill_between(leadlags,mu-sigma,mu+sigma,color="k",label="",alpha=0.10,zorder=-1)


for ex in range(nexps):
    if plothp:
        threscorr = ccfs_all_hp[ex,:,:]
    else:
        threscorr = ccfs_all[ex,:,:] # 
    mu        = threscorr.mean(0)
    ax.plot(leadlags,mu,label=expnames[ex],c=expcols[ex],ls=expls[ex],lw=2.5)

ax.set_xticks(xtks)
ax = viz.add_ticks(ax)
ax.set_xlim([xtks[0],xtks[-1]])

ax.legend(ncol=2)

ax.set_ylabel("Correlation")
ax.set_xlabel("<--- SSS Leads SST | SSS Lags SST --->")
ax.set_ylim([-1,1])

ax.axvline([0],lw=0.55,c="k",zorder=-3)
ax.axhline([0],lw=0.55,c="k",zorder=-3)
ax.set_title("SST-SSS Correlation at 50N, 30W (Stochastic model, White Noise and T' Coupling)")

#savename = "%sSPG_Point_SST_SSS_CrossCorr_Hipass%02i_CESMvSM_%s_alpha%i.png" % (figpath,hicutoff,expname,add_alpha)

# if zoom:
#     ax.set_ylim([-.05,.05])
#     savename = proc.addstrtoext(savename,"_zoom")

#plt.savefig(savename,dpi=200,)

            
            