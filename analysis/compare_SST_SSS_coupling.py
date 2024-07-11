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

# Indicate point where calculation was performed
lonf = -30
latf = 50

"""
SST-SSS, Effect of White Noise and Coupling
(1) 

"""
# Paths and Experiment
comparename  = "NoiseCoupling"
expnames     = ['SST_SSS_LHFLX','SST_SSS_LHFLX_1','SST_SSS_LHFLX_2','SST_SSS_LHFLX_2_neg',
                'SST_SSS_LHFLX_DiffWn','SST_SSS_LHFLX_DiffT',"SST_SSS_LHFLX_DiffAll",'SST_SSS_LHFLX_2_noLbdE']
expcols      = ["k","gray","goldenrod",'orange',"blue","salmon","magenta",'cyan']
expls        = ["solid",'dashed','dotted','dashed','dotted',"dashed","dotted",'solid']


# -----------------------------------------------------------------------------

"""
SSS, Effect of adding precipitation, Qek, etc
"""
comparename  = "SSS_LHFLX_Hierarchy"
expnames     = ["SSS_LHFLX_only","SSS_LHFLX_addP","SSS_LHFLX_addQek",
                "SSS_LHFLX_addP_addQek","SST_SSS_NHFLX"]

expcols      = ["gray","cornflowerblue","orange",'hotpink',"navy",]
expls        = ["solid",'dashed','dotted','solid','dashed',]


# metrics_path = output_path + expname + "/Metrics/" 
# exp_output   = output_path + expname + "/Output/" 

vnames       = ["SST","SSS",]
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

#%% Load CESM1 ACFs for comparison

cesm_nc_sss = procpath + "CESM1_1920to2005_SSSACF_lag00to60_ALL_ensALL.nc"
cesm_nc_sst = procpath + "CESM1_1920to2005_SSTACF_lag00to60_ALL_ensALL.nc"

ds_cesm_sss = xr.open_dataset(cesm_nc_sss).acf.load()
ds_cesm_sst = xr.open_dataset(cesm_nc_sst).acf.load()

cesm_acfs = [ds_cesm_sst,ds_cesm_sss,]
cesm_acfs_pt = [proc.selpt_ds(ds,lonf,latf).mean('ens').squeeze() for ds in cesm_acfs]


#%% Load in CESM SST/SSS for cross-correlation computation

cesm_vars = []
for vv in range(2):
    ncname = "%sCESM1LE_%s_NAtl_19200101_20050101_bilinear.nc" % (rawpath,vnames[vv])
    ds     = xr.open_dataset(ncname).sel(lon=lonf,lat=latf,method='nearest')[vnames[vv]].load()
    cesm_vars.append(ds)

# Deseasonalize, Anomalize
def preproc_cesm(ds):
    dsdt = ds - ds.mean('ensemble')
    dsda = proc.xrdeseason(dsdt)
    dsda = dsda.rename({"ensemble":"ens"})
    return dsda

cesm_vanom = [preproc_cesm(ds) for ds in cesm_vars]


#%% Load the variables (stochastic model)
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
vv      = 1
kmonth  = 6
xtks    = np.arange(37)
mons3 = proc.get_monstr()

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
        
    
    # Plot CESM
    cesmacf = cesm_acfs_pt[vv].isel(mons=kmonth)
    ax.plot(cesmacf.lags,cesmacf,label="CESM1 (Ens. Avg.)",c="k",ls="solid",lw=2.5)
    #cesm_acfs_pt = [proc.selpt_ds(ds,lonf,latf).mean('ens').squeeze() for ds in cesm_acfs]

    ax.legend(ncol=2)
    
    ax.set_title("%s Monthly Autocorrelation Function (Lag 0 = %s)" % (vnames[vv],mons3[kmonth]),fontsize=16)
    savename = "%s%s_ACF_basemon%02i.png" % (figpath,comparename,kmonth+1)
    plt.savefig(savename,dpi=150,bbox_inches='tight')
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
        
#%% Also compute cross correlation for CESm1 (copied from viz_SST_SSS_coupling)

# Here is the xarray ufunc version
calc_leadlag    = lambda x,y: proc.leadlag_corr(x,y,lags,corr_only=True)
llcesm = xr.apply_ufunc(
    calc_leadlag,
    cesm_vanom[0],
    cesm_vanom[1],
    input_core_dims=[['time'],['time']],
    output_core_dims=[['lags']],
    vectorize=True,
    )
leadlags     = np.concatenate([np.flip(-1*lags)[:-1],lags],) 
llcesm['lags'] = leadlags

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


# Plot CESM1
ax.plot(llcesm.lags,llcesm.mean('ens'),color="k",label="CESM (Ens. Avg.)",lw=2.5)
for e in range(42):
    ax.plot(llcesm.lags,llcesm.isel(ens=e),color="gray",label="",lw=1.5,alpha=0.2,zorder=-1)
    

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


savename = "%s%s_CCF.png" % (figpath,comparename)
plt.savefig(savename,dpi=150,bbox_inches='tight')
    

#%% Check how monthly variance is impacted

vv         = 1
nrun,ntime = var_byexp[ex][vv].shape
nyrs       = int(ntime/12)

vunits     = ["$\degreeC$","$psu$"]

fig,ax     = viz.init_monplot(1,1,figsize=(8,4.5))

for ex in range(nexps):
    
    # Compute Monthly Standard Deviation
    varin = var_byexp[ex][vv].reshape(nruns,nyrs,12).std(1)
    mu    = varin.mean(0)
    
    # Plot it
    ax.plot(mons3,mu,label=expnames[ex],c=expcols[ex],ls=expls[ex],lw=2.5)

# Plot for CESM
ntime_cesm = int(1032/12)
cesmvarin  = np.nanstd(cesm_vanom[vv].values.reshape(42,ntime_cesm,12),1)
mu         = cesmvarin.mean(0)

ax.plot(mons3,mu,color="k",label="CESM (Ens. Avg.)",lw=2.5)
for e in range(42):
    ax.plot(mons3,cesmvarin[e,:],color="gray",label="",lw=1.5,alpha=0.2,zorder=-1)
    

ax.legend(fontsize=8,ncol=4)
ax.set_title("Monthly Stdev. (%s)" % (vnames[vv]))
ax.set_ylabel(vunits[vv])

savename = "%s%s_Monvar.png" % (figpath,comparename)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Compuate and look at the power spectra (may adapt for larger script)!

# Compute the power spectra (Testbed function from xrfunc)
def pointwise_spectra(tsens,nsmooth=1, opt=1, dt=None, clvl=[.95], pct=0.1):
    calc_spectra = lambda x: proc.point_spectra(x,nsmooth=nsmooth,opt=opt,
                                                dt=dt,clvl=clvl,pct=pct)
    
    # Change NaN to Zeros for now
    tsens_nonan = xr.where(np.isnan(tsens),0,tsens)
    
    # Compute Spectra
    specens = xr.apply_ufunc(
        proc.point_spectra,  # Pass the function
        tsens_nonan,  # The inputs in order that is expected
        # Which dimensions to operate over for each argument...
        input_core_dims=[['time'],],
        output_core_dims=[['freq'],],  # Output Dimension
        exclude_dims=set(("freq",)),
        vectorize=True,  # True to loop over non-core dims
    )
    
    # # Need to Reassign Freq as this dimension is not recorded
    # ts1  = tsens.isel(ens=0).values
    # freq = proc.get_freqdim(ts1)
    # specens['freq'] = freq
    return specens

dtplot   = 3600*24*365
# Do computation for stochastic model ----------------------------------------
# Separate by variable and take the annual average...
sm_sst = [proc.ann_avg(var_byexp[ex][0],1) for ex in range(nexps)] # [Run x Time]
sm_sss = [proc.ann_avg(var_byexp[ex][1],1) for ex in range(nexps)] # [Run x Time]

# Combine Arrays to apply ufunc
smyrs     = xr.cftime_range(start='0001',periods=sm_sst[0].shape[1],freq="YS",calendar="noleap")
sm_inputs = np.array([np.array(sm_sst),np.array(sm_sss)]) # [Variable x Experiment x Run x Year]
coords    = dict(var=vnames,exp=expnames,ens=np.arange(10),time=smyrs)
da_sm     = xr.DataArray(sm_inputs,coords=coords,dims=coords)

# Compute Spectra (using ufunc)
nsmooth_sm      = 50
sm_spec         = pointwise_spectra(da_sm,nsmooth=nsmooth_sm,dt=dtplot)
freq            = proc.get_freqdim(da_sm.isel(var=0,ens=0,exp=0).values)
sm_spec['freq'] = freq


# Do computation for CESM ----------------------------------------------------
# Repeat for CESM, but with different nsmooth
da_cesm        = xr.concat(cesm_vanom,dim='var')
da_cesm['var'] = vnames
nsmooth_cesm   = 5
cesm_spec      = pointwise_spectra(da_cesm,nsmooth=nsmooth_cesm,dt=dtplot)


#%% Alternative, just go a silly loop with quick spectrum

#%% Now compare the spectra

# Select what to plot
vv       = 0
#dtplot   = 3600*24*30

fig,ax   = plt.subplots(1,1,constrained_layout=True,figsize=(10,4.5))#,sharey=True)

toplab=True
botlab=True

ax       = viz.init_logspec(1,1,ax=ax,toplab=toplab,botlab=botlab,dtplot=dtplot)
#ax.set_title(regions_long[rr],fontsize=22)

# Plot for each experiment (stochastic model)
for ex in range(nexps):
    
    #svarsin = specexp[ex][rr]
    
    P     = sm_spec.isel(var=vv,exp=ex).mean('ens')
    #svarsin['specs']
    freq  = sm_spec.freq#svarsin['freqs']
    
    #cflab = "Red Noise"
    #CCs   = svarsin['CCs']
    
    # Convert units
    freq     = freq * dtplot
    P        = P / dtplot
    #Cbase    = CCs.mean(0)[:, 0]/dtplot
    #Cupbound = CCs.mean(0)[:, 1]/dtplot
    #
    # Plot Ens Mean
    #mu    = P.mean(0)
    #sigma = P.std(0)
    
    # # Plot Spectra
    ax.loglog(freq, P, c=expcols[ex], lw=2.5,
            label=expnames[ex],)
    
    # # Plot Significance
    # if ex ==0:
    #     labc1 = cflab
    #     labc2 = "95% Confidence"
    # else:
    #     labc1=""
    #     labc2=""
    # ax.plot(freq, Cbase, color=ecols[ex], ls='solid', lw=1.2, label=labc1)
    # ax.plot(freq, Cupbound, color=ecols[ex], ls="dotted",
    #         lw=2, label=labc2)
# if rr == 0:
#     ax.legend(ncol=2)
    
# if comparename ==  "lbde_comparison_CSU":
#     ax.set_ylim([1e-4,1e-1])
#     vunit = "psu"
# else:
#     vunit = "$\degree$C"

#ax.set_ylabel("Power (%s$^2$/cpy)" % vunit,fontsize=fsz_axis)

#%%



#savename = "%sSPG_Point_SST_SSS_CrossCorr_Hipass%02i_CESMvSM_%s_alpha%i.png" % (figpath,hicutoff,expname,add_alpha)

# if zoom:
#     ax.set_ylim([-.05,.05])
#     savename = proc.addstrtoext(savename,"_zoom")

#plt.savefig(savename,dpi=200,)

            
            