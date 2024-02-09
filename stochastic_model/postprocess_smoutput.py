#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Postprocess Stochastic Model Output from SSS basinwide Integrations

Currently a working draft, will copy the essential functions once I have finalized things

Created on Wed Feb  7 17:23:00 2024

@author: gliu

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr
import sys
from tqdm import tqdm
import copy
import glob

#%% Import Custom Modules
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

sys.path.append(scmpath + '../')

from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx

import stochmod_params as sparams



#%% Get Experiment Information

expname = "SSS_OSM_Tddamp"
varname = "SSS"


#%% Load output (copied from analyze_basinwide_output_SSS)

output_path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/sm_experiments/"
figpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/02_Figures/20240209/"
proc.makedir(figpath)

# Load NC Files
expdir       = output_path + expname + "/Output/"
nclist       = glob.glob(expdir +"*.nc")
nclist.sort()
print(nclist)

# Load DS, deseason and detrend to be sure
ds_all   = xr.open_mfdataset(nclist,concat_dim="run",combine='nested').load()

ds_sm  = proc.xrdeseason(ds_all[varname])
ds_sm  = ds_sm - ds_sm.mean('run')
ds_sm  = ds_sm.rename(dict(run='ens'))

# Load Param Dictionary
dictpath   = output_path + expname + "/Input/expparams.npz"
expdict  = np.load(dictpath,allow_pickle=True)


#%% Load CESM1 Output for SSS (copied from analyze_basinwide_output_SSS)

# Loading old anomalies
#ncpath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/CESM_proc/"
#ncname  = "%s_FULL_HTR_lon-80to0_lat0to65_DTEnsAvg.nc" % varname
#anom_cesm = True

# Loading anomalies used in recent scripts (find origin, I think its prep_var_monthly, etc)
ncpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"
ncname    = "CESM1LE_%s_NAtl_19200101_20050101_bilinear.nc" % varname
anom_cesm = False # Set to false to anomalize

# Load DS
ds_cesm  = xr.open_dataset(ncpath+ncname).squeeze()

# Slice to region
bbox_sim = expdict['bbox_sim']
ds_cesm  = proc.sel_region_xr(ds_cesm,bbox_sim)

# Correct Start time
ds_cesm  = proc.fix_febstart(ds_cesm)
ds_cesm  = ds_cesm.sel(time=slice('1920-01-01','2005-12-31')).load()

# Anomalize if necessary
if anom_cesm is False:
    print("Detrending and deseasonalizing variable!")
    ds_cesm = proc.xrdeseason(ds_cesm) # Deseason
    ds_cesm = ds_cesm[varname] - ds_cesm[varname].mean('ensemble')
else:
    ds_cesm = ds_cesm[varname]

# Rename ens
ds_cesm = ds_cesm.rename(dict(ensemble='ens'))

#%% Load some dims for plotting (copied from analyze_basinwide_output_SSS)
ds             = ds_all
lon            = ds.lon.values
lat            = ds.lat.values
tsim           = ds.time.values

latf           = 50
lonf           = -30
klon,klat      = proc.find_latlon(lonf,latf,lon,lat)
locfn,loctitle = proc.make_locstring(lonf,latf)

#%% Set up for processing

ds_in  = [ds_sm,ds_cesm]
enames = ["Stochastic Model","CESM1"]
cols   = ["orange","k"]
lss    = ['dashed','solid']
mks    = ["d","o"]

[print(ds.shape) for ds in ds_in]

# Make a mask
msk = ds_in[0].mean('time').mean('ens').values
msk[~np.isnan(msk)] = 1

# ---------------------------------------------------
#%% Part (1) Overall Variance
# ---------------------------------------------------

dsvar_byens    = [ds.std('time') * msk for ds in ds_in]
dsvar_seasonal = [ds.groupby('time.season').std('time') for ds in ds_in]


#%% Plotting parameters

bbplot                      = [-80,0,15,65]
mpl.rcParams['font.family'] = 'Avenir'

fsz_title=20

#%% First, calculate monthly variance, and plot

vmax   = 0.75
pmesh  = False
if varname == "SST":
    levels = np.arange(0,1,0.1)
else:
    levels = np.arange(0,0.30,0.02)
fig,axs,mdict = viz.init_orthomap(1,2,bbplot,figsize=(10,4.5))

for a,ax in enumerate(axs):
    
    ax   = viz.add_coast_grid(ax,bbox=bbplot,fill_color='lightgray')
    pv   = dsvar_byens[a].mean('ens') * msk
    if pmesh:
        pcm  = ax.pcolormesh(pv.lon,pv.lat,pv,transform=mdict['noProj'],vmin=0,vmax=vmax)
    else:
        pcm  = ax.contourf(pv.lon,pv.lat,pv,transform=mdict['noProj'],levels=levels,extend="both")

    ax.set_title(enames[a],fontsize=fsz_title)
    

cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.05,pad=0.02)
cb.set_label("$\sigma$ (%s)" % varname)

savename = "%s%s_Overall_Variance_Comparison" % (figpath,expname,)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Zonal and Meridional Averages




#%% Examine the seasonal variances

fsz_title = 20
fsz_axis  = 18
cmap      = 'cmo.thermal'

fig,axs,mdict = viz.init_orthomap(2,4,bbplot,figsize=(16,7))

for ee in range(2):
    
    for s in range(4):
        
        ax   = axs[ee,s]
        ax   = viz.add_coast_grid(ax,bbox=bbplot,fill_color="k")
        pv   = dsvar_seasonal[ee].mean('ens').isel(season=s) * msk
        if pmesh:
            pcm  = ax.pcolormesh(pv.lon,pv.lat,pv,transform=mdict['noProj'],vmin=0,vmax=vmax,cmap=cmap)
        else:
            pcm  = ax.contourf(pv.lon,pv.lat,pv,transform=mdict['noProj'],levels=levels,extend="both",cmap=cmap)
        
        if s == 0:
            viz.add_ylabel(enames[ee],ax=ax,x=-.15,fontsize=fsz_axis)
        if ee == 0:
            ax.set_title(pv.season.values,fontsize=fsz_title)
            
cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.05,pad=0.04)
cb.set_label("$\sigma$ (%s)" % varname)
savename = "%s%s_Seasonal_Variance_Comparison" % (figpath,expname,)
plt.savefig(savename,dpi=150,bbox_inches='tight')

# ---------------------------------------------------
#%% Part (2) Regional Analysis
# ---------------------------------------------------

#% Do some regional analysis
bbxall      = sparams.bboxes
regionsall  = sparams.regions 
rcolsall    = sparams.bbcol

# Select Regions
regions_sel = ["SPG","NNAT","STGe","STGw"]
bboxes      = [bbxall[regionsall.index(r)] for r in regions_sel]
rcols       = [rcolsall[regionsall.index(r)] for r in regions_sel]

#%% Compute Regional Averages

tsm_regs = []
ssts_reg = []
for r in range(len(regions_sel)):
    #rshapes = []
    
    # Take the area weighted average
    bbin  = bboxes[r]
    rssts = [proc.sel_region_xr(ds,bbin) for ds in ds_in]
    ravgs = [proc.area_avg_cosweight(ds) for ds in rssts]
    
    # Compute some metrics
    rsstin = [rsst.values.flatten() for rsst in ravgs]
    rsstin = [np.where((np.abs(rsst)==np.inf) | np.isnan(rsst),0.,rsst) for rsst in rsstin]
    tsmr   = scm.compute_sm_metrics(rsstin)
    
    tsm_regs.append(tsmr)
    ssts_reg.append(ravgs)
    
    
#%% Make Some Plots (ACFs)
nregs = len(bboxes)

kmonth = 7
xtksl  = np.arange(0,37,3)
lags   = np.arange(37)

fig,axs = plt.subplots(2,2,constrained_layout=True,figsize=(16,8))

for rr in range(nregs):
    
    ax   = axs.flatten()[rr]
    ax,_ = viz.init_acplot(kmonth,xtksl,lags,title="",ax=ax,fsz_axis=fsz_axis,fsz_ticks=14)
    
    for ii in range(2):
        plotvar = tsm_regs[rr]['acfs'][kmonth][ii]
        ax.plot(lags,plotvar,label=enames[ii],c=cols[ii],ls=lss[ii],marker=mks[ii])
    ax.legend()
    ax.set_title(regions_sel[rr],fontsize=fsz_title)
    
savename = "%s%s_Regional_ACF_Comparison_mon%02i.png" % (figpath,expname,kmonth+1)
plt.savefig(savename,dpi=150,bbox_inches='tight')
    
    
#%% Plot Monthly Varaiance

mons3=proc.get_monstr()

fig,axs = plt.subplots(2,2,constrained_layout=True,figsize=(10,6.5))

for rr in range(nregs):
    
    ax   = axs.flatten()[rr]
    
    #ax = viz.add_ticks(ax=ax,)
    #ax,_ = viz.init_monplot()
    
    for ii in range(2):
        plotvar = tsm_regs[rr]['monvars'][ii]
        plotlab = "%s (var=%.2e $psu^2$)" % (enames[ii],np.var(ssts_reg[rr][ii]))
        ax.plot(mons3,plotvar,label=plotlab,c=cols[ii],ls=lss[ii],marker=mks[ii])
    ax.legend()
    
    ax.set_title(regions_sel[rr],fontsize=fsz_title)

savename = "%s%s_Regional_MonthlyVariance_Comparison.png" % (figpath,expname)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Plot Regional Spectra

# Plotting Params
dtplot  = 3600*24*30
plotyrs = [100,50,25,10,5]
xtks    = 1/(np.array(plotyrs)*12)


fig,axs = plt.subplots(2,2,constrained_layout=True,figsize=(10,6.5))


for rr in range(nregs):
    
    ax   = axs.flatten()[rr]
    
    #ax = viz.add_ticks(ax=ax,)
    #ax,_ = viz.init_monplot()
    
    for ii in range(2):
        plotfreq = tsm_regs[rr]['freqs'][ii] * dtplot
        plotspec = tsm_regs[rr]['specs'][ii] / dtplot #tsm_regs[rr]['monvars'][ii]
        plotlab = "%s (var=%.3f $psu^2$)" % (enames[ii],np.var(ssts_reg[rr][ii]))
        ax.plot(plotfreq,plotspec,label=plotlab,c=cols[ii],ls=lss[ii])
    ax.legend()
    
    ax.set_xticks(xtks,labels=plotyrs)
    ax.set_xlim([xtks[0],xtks[-1]])
    
    ax.set_xlabel("Period")
    ax.set_ylabel("Power ($psu^2$ / cpy)")
    ax.set_title(regions_sel[rr],fontsize=fsz_title)
savename = "%s%s_Regional_Spectra.png" % (figpath,expname)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Look at actual timeseries

irun = 8
fig,axs = plt.subplots(4,1,constrained_layout=True,figsize=(16,14))

for rr in range(nregs):
    
    ax   = axs.flatten()[rr]
    
    #ax = viz.add_ticks(ax=ax,)
    #ax,_ = viz.init_monplot()
    
    for run in range(10):
        plotvar = ssts_reg[rr][0][run,:]
        ax.plot(plotvar,alpha=0.7,label="Run %02i" % (run+1))
        
    ax.plot(ssts_reg[rr][0][:,:].isel(ens=irun),lw=0.75,
            alpha=1,label="Run %i" % (irun+1),c="k")
    if rr == 0:
        ax.legend(ncol=5)
    ax.set_title(regions_sel[rr],fontsize=fsz_title)
    
savename = "%s%s_Regional_Timeseries.png" % (figpath,expname)
plt.savefig(savename,dpi=150,bbox_inches='tight')


# ---------------------------------------------------
#%% Part (3) Pointwise Autocorrelation
# ---------------------------------------------------

pathout = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
ncout = "SM_SST_OSM_Tddamp_SST_autocorrelation_thresALL_lag00to60.nc"

dsout = xr.open_dataset(pathout + ncout)

acfsout = dsout.SST.isel(thres=0)

T2out = proc.calc_T2(acfsout,axis=-1)

#%% scrap/WIP
#%%



# Based on pointwise_autocorrelation from stochmod/analysis
lags        = np.arange(0,61,1)
lagname     = "lag%02ito%02i" % (lags[0],lags[-1]) 
conf        = 0.95
tails       = 2

ds_in_sm    = ds_in[0]

# First step: transpose the dimensions
invar      = ds_in_sm.transpose('lon','lat','ens','time').values



def calc_acf_pointwise(invar,lags,):
    
    # Get Dimensions
    nlon,nlat,nens,ntime = invar.shape
    npts                 = nlon*nlat*nens
    nlags                = len(lags)
    
    # Reshape and locate non NaN points
    invar_rs = invar.reshape(npts,ntime)
    nandict  = proc.find_nan(invar_rs,1,return_dict=True)
    validpts = nandict['cleaned_data']
    npts_valid,_ = validpts.shape
    
    
    # Split to year x mon
    validpts = validpts.reshape(npts_valid)
    
#%%

def calc_acf_pointwise(invar,lags,thresholds=[0,],thresvar=None):
    
    # Get Dimensions
    nlon,nlat,nens,ntime = invar.shape
    npts                 = nlon*nlat*nens
    nlags                = len(lags)
    
    nyr             = int(ntime/12)
    nlags           = len(lags)
    nthres          = len(thresholds)

    # Combine space, remove NaN points
    sstrs                = invar.reshape(npts,ntime)
    if np.all(np.isnan(sstrs.sum(1))): # At least 1 point should be valid for all time...
        print("Warning, something may be corrupted along the time dimension...")
        exit
    
    
    if thresvar: # Only analyze where both threshold variable and target var are non-NaN
        loadvarrs     = thresvar.reshape(npts,ntime)
        _,knan,okpts  = proc.find_nan(sstrs*loadvarrs,1) # [finepoints,time]
        sst_valid     = sstrs[okpts,:]
        loadvar_valid = loadvarrs[okpts,:]
    else:
        sst_valid,knan,okpts = proc.find_nan(sstrs,1) # [finepoints,time]
    npts_valid           = sst_valid.shape[0] 


    # Split to Year x Month
    sst_valid = sst_valid.reshape(npts_valid,nyr,12)
    if thresvar: # Select non-NaN points for thresholding variable
        loadvar_valid = loadvar_valid.reshape(npts_valid,nyr,12)

    # Preallocate (nthres + 1 (for all thresholds), and last is all data)
    class_count = np.zeros((npts_valid,12,nthres+2)) # [pt x eventmonth x threshold]
    sst_acs     = np.zeros((npts_valid,12,nthres+2,nlags))  # [pt x eventmonth x threshold x lag]
    sst_cfs     = np.zeros((npts_valid,12,nthres+2,nlags,2))  # [pt x eventmonth x threshold x lag x bounds]
    
    # A pretty ugly loop....
    # Now loop for each month
    for im in range(12):
        #print(im)
        
        # For that month, determine which years fall into which thresholds [pts,years]
        sst_mon = sst_valid[:,:,im] # [pts x yr]
        if thresvar:
            loadvar_mon = loadvar_valid[:,:,im]
            sst_mon_classes = proc.make_classes_nd(loadvar_mon,thresholds,dim=1,debug=False)
        else:
            sst_mon_classes = proc.make_classes_nd(sst_mon,thresholds,dim=1,debug=False)
        
        for th in range(nthres+2): # Loop for each threshold
        
            if th < nthres + 1: # Calculate/Loop for all points
                for pt in tqdm(range(npts_valid)): 
                    
                    # Get years which fulfill criteria
                    yr_mask     = np.where(sst_mon_classes[pt,:] == th)[0] # Indices of valid years
                    
                    
                    #sst_in      = sst_valid[pt,yr_mask,:] # [year,month]
                    #sst_in      = sst_in.T
                    #class_count[pt,im,th] = len(yr_mask) # Record # of events 
                    #ac = proc.calc_lagcovar(sst_in,sst_in,lags,im+1,0) # [lags]
                    
                    # Compute the lagcovariance (with detrending)
                    sst_in = sst_valid[pt,:,:].T # transpose to [month x year]
                    ac,yr_count = proc.calc_lagcovar(sst_in,sst_in,lags,im+1,0,yr_mask=yr_mask,debug=False)
                    cf = proc.calc_conflag(ac,conf,tails,len(yr_mask)) # [lags, cf]
                    
                    # Save to larger variable
                    class_count[pt,im,th] = yr_count
                    sst_acs[pt,im,th,:] = ac.copy()
                    sst_cfs[pt,im,th,:,:]  = cf.copy()
                    # End Loop Point -----------------------------
            
            
            else: # Use all Data
                print("Now computing for all data on loop %i"%th)
                # Reshape to [month x yr x npts]
                sst_in    = sst_valid.transpose(2,1,0)
                acs = proc.calc_lagcovar_nd(sst_in,sst_in,lags,im+1,1) # [lag, npts]
                cfs = proc.calc_conflag(acs,conf,tails,nyr) # [lag x conf x npts]
                
                # Save to larger variable
                sst_acs[:,im,th,:] = acs.T.copy()
                sst_cfs[:,im,th,:,:]  = cfs.transpose(2,0,1).copy()
                class_count[:,im,th]   = nyr
            # End Loop Threshold -----------------------------
        
    # End Loop Event Month -----------------------------

    #% Now Replace into original matrices
    # Preallocate
    count_final = np.zeros((npts,12,nthres+2)) * np.nan
    acs_final   = np.zeros((npts,12,nthres+2,nlags)) * np.nan
    cfs_final   = np.zeros((npts,12,nthres+2,nlags,2)) * np.nan

    # Replace
    count_final[okpts,...] = class_count
    acs_final[okpts,...]   = sst_acs
    cfs_final[okpts,...]   = sst_cfs

    # Reshape output
    if notherdims == 0:
        count_final = count_final.reshape(nlon,nlat,12,nthres+2)
        acs_final   = acs_final.reshape(nlon,nlat,12,nthres+2,nlags)
        cfs_final   = cfs_final.reshape(nlon,nlat,12,nthres+2,nlags,2)
    else:
        count_final = count_final.reshape(nlon,nlat,notherdims,12,nthres+2)
        acs_final   = acs_final.reshape(nlon,nlat,notherdims,12,nthres+2,nlags)
        cfs_final   = cfs_final.reshape(nlon,nlat,notherdims,12,nthres+2,nlags,2)

# Get Threshold Labels
threslabs   = []
if nthres == 1:
    threslabs.append("$T'$ <= %i"% thresholds[0])
    threslabs.append("$T'$ > %i" % thresholds[0])
else:
    for th in range(nthres):
        thval= thresholds[th]
        
        if thval != 0:
            sig = ""
        else:
            sig = "$\sigma$"
        
        if th == 0:
            tstr = "$T'$ <= %i %s" % (thval,sig)
        elif th == nthres:
            tstr = "$T'$ > %i %s" % (thval,sig)
        else:
            tstr = "%i < $T'$ =< %i %s" % (thresholds[th-1],thval,sig)
        threslabs.append(th)
threslabs.append("ALL")

#% Save Output
np.savez(savename,**{
    'class_count' : count_final,
    'acs' : acs_final,
    'cfs' : cfs_final,
    'thresholds' : thresholds,
    'lon' : lon,
    'lat' : lat,
    'lags': lags,
    'threslabs' : threslabs
    },allow_pickle=True)

    
    
    

