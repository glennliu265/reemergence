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
import time

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

#expname = "SST_EOF_Qek_pilot"
#varname = "SST"

# expname = "SSS_EOF_Qek_pilot_corrected"
# varname = "SSS"

expname   = "SSS_EOF_Qek_LbddEnsMean"
varname   = "SSS"

#%% Load output (copied from analyze_basinwide_output_SSS)

st = time.time()
output_path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/sm_experiments/"
figpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20240308/"
proc.makedir(figpath)


# Load NC Files
expdir      = output_path + expname + "/Output/"
expmetrics  = output_path + expname + "/Metrics/"
nclist      = glob.glob(expdir +"*.nc")
nclist.sort()

# Load DS, deseason and detrend to be sure
ds_all      = xr.open_mfdataset(nclist,concat_dim="run",combine='nested').load()

ds_sm       = proc.xrdeseason(ds_all[varname])
ds_sm       = ds_sm - ds_sm.mean('run')
ds_sm       = ds_sm.rename(dict(run='ens'))

# Load Param Dictionary
dictpath    = output_path + expname + "/Input/expparams.npz"
expdict     = np.load(dictpath,allow_pickle=True)

print("Output loaded in %.2fs" % (time.time()-st))

# ---------------------------------------------
#%% Load the BSF and SSH, get seasonal averages
# ---------------------------------------------

ds_bsf      = dl.load_bsf(ensavg=True)
ds_ssh      = dl.load_bsf(ensavg=True,ssh=True)

long        = ds_bsf.lon
latg        = ds_bsf.lat

ytime       = proc.get_xryear()

# Take seasonal averages
dscurr      = [ds_bsf,ds_ssh]
dscurr_savg = [proc.calc_savg(ds,ds=True) for ds in dscurr] 

#%% Do a test plot of a point

irun      = 0
lonf,latf = -30,55
ts        = ds_sm[irun].sel(lon=lonf,lat=latf,method='nearest').values
plt.plot(ts),plt.show()

#%% Load CESM1 Output for SSS (copied from analyze_basinwide_output_SSS)

st = time.time()
# Loading old anomalies
#ncpath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/CESM_proc/"
#ncname  = "%s_FULL_HTR_lon-80to0_lat0to65_DTEnsAvg.nc" % varname
#anom_cesm = True

# Loading anomalies used in recent scripts (find origin, I think its prep_var_monthly, etc)
ncpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"
ncname    = "CESM1LE_%s_NAtl_19200101_20050101_bilinear.nc" % varname
anom_cesm = False # Set to false to anomalize

# Load DS
ds_cesm   = xr.open_dataset(ncpath+ncname).squeeze()

# Slice to region
bbox_sim  = expdict['bbox_sim']
ds_cesm   = proc.sel_region_xr(ds_cesm,bbox_sim)

# Correct Start time
ds_cesm   = proc.fix_febstart(ds_cesm)
ds_cesm   = ds_cesm.sel(time=slice('1920-01-01','2005-12-31')).load()

# Anomalize if necessary
if anom_cesm is False:
    print("Detrending and deseasonalizing variable!")
    ds_cesm = proc.xrdeseason(ds_cesm) # Deseason
    ds_cesm = ds_cesm[varname] - ds_cesm[varname].mean('ensemble')
else:
    ds_cesm = ds_cesm[varname]

# Rename ens
ds_cesm = ds_cesm.rename(dict(ensemble='ens'))

print("Loaded CESM output in %.2fs" % (time.time()-st))

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

dsvar_byens    = [ds.std('time')  for ds in ds_in]
dsvar_seasonal = [ds.groupby('time.season').std('time') for ds in ds_in]

#%% Plotting parameters

bbplot                      = [-80,0,20,65]
mpl.rcParams['font.family'] = 'Avenir'

fsz_title                   = 20

#%% Load Land Ice Mask (Copied from visualize_acf_rmse)

# Set names for land ice mask (this is manual, and works just on Astraeus :(...!)
lipath          = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/Masks/"
liname          = "CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc"

# Load Land Ice Mask
ds_mask          = xr.open_dataset(lipath+liname).MASK.squeeze().load()

# Edit
plotmask = ds_mask.values.copy()
plotmask[np.isnan(plotmask)] = 0.

maskcoast = ds_mask.values.copy()
maskcoast = np.roll(maskcoast,1,axis=0) * np.roll(maskcoast,-1,axis=0) * np.roll(maskcoast,1,axis=1) * np.roll(maskcoast,-1,axis=1)

#%% First, calculate monthly variance, and plot

#iens        = 1
vmax          = None#.2
pmesh         = False
slvls         = np.arange(-150,160,15)
#cmap          = "cmo.thermal"

if varname == "SST":
    levels = np.arange(0,1,0.1)
    vunit  = "$\degree C$"
else:
    levels = np.arange(0,0.24,0.02)
    vunit  = "$psu$"
fig,axs,mdict = viz.init_orthomap(1,2,bbplot,figsize=(10,4.5))

for a,ax in enumerate(axs):
    
    ax   = viz.add_coast_grid(ax,bbox=bbplot,fill_color='lightgray')
    pv   = dsvar_byens[a].mean('ens') * msk
    
    if vmax is None:
        #pv   = dsvar_byens[a].isel(ens=iens) #* msk
        if pmesh:
            pcm  = ax.pcolormesh(pv.lon,pv.lat,pv,transform=mdict['noProj'],vmin=0,vmax=vmax)
        else:
            pcm  = ax.contourf(pv.lon,pv.lat,pv,transform=mdict['noProj'],levels=levels,extend="both")

    else:
        if pmesh:
            pcm  = ax.pcolormesh(pv.lon,pv.lat,pv,transform=mdict['noProj'])
           
        else:
            pcm  = ax.contourf(pv.lon,pv.lat,pv,transform=mdict['noProj'],extend="both")
        fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.05,pad=0.01)
    ax.set_title(enames[a],fontsize=fsz_title)
    
    
    # Plot contours    
    current = dscurr[1].mean('mon').SSH
    cl = ax.contour(long,latg,current,colors="k",
                    linewidths=0.35,transform=mdict['noProj'],levels=slvls,alpha=0.8)
    ax.clabel(cl)
    
    #cl = ax.contour(pv.lon,pv.lat,msk,levels=[0,1,2],colors="w",transform=mdict['noProj'])
    
    # Plot Mask
    cl2 = ax.contour(ds_mask.lon,ds_mask.lat,plotmask,colors="w",linestyles='dashed',linewidths=.95,
                    levels=[0,1],transform=mdict['noProj'],zorder=1)
    

cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.05,pad=0.02)
cb.set_label("$\sigma$ (%s, %s)" % (varname,vunit))

savename = "%s%s_Overall_Variance_Comparison.png" % (figpath,expname,)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Same as above, but plot the difference and ratio

cmap_diff     = 'cmo.balance'
slvls         = np.arange(-150,160,15)
pmesh         = False

# Initialize Figure
fig,axs,mdict = viz.init_orthomap(1,2,bbplot,figsize=(10,4.5))

for a in range(2):
    
    ax = axs[a]
    ax   = viz.add_coast_grid(ax,bbox=bbplot,fill_color='lightgray')
    
    if a == 0:
        pv     = (dsvar_byens[0].mean('ens') - dsvar_byens[1].mean('ens')) * msk 
        title  = "Diff. ($\sigma_{SM} - \sigma_{CESM}$)"
        
        if varname == 'SST':
            vlm    = [-.5,.5]
            vlvls  = np.arange(-.5,.55,0.05)
        elif varname == 'SSS':
            vlm    = [-.3,.3]
            vlvls  = np.arange(-.3,.33,0.03)
            
        #cblab  = "$\sigma_{SM} - \sigma_{CESM}$"
    elif a == 1:
        pv     = np.log(dsvar_byens[0].mean('ens')/dsvar_byens[1].mean('ens')) * msk
        title  = "Log($\sigma_{SM}/\sigma_{CESM}$)"
        if varname == 'SST':
            vlm    = [-1,1]
            vlvls  = np.arange(-1,1.1,0.1)
        elif varname == 'SSS':
            vlm    = [-2.3,2.3]
            vlvls  = np.arange(-2.5,2.75,0.25)
        
        #cblab  = "Log($\sigma_{SM}/\sigma_{CESM}$)"
    # Plot the values
    ax.set_title(title)
    if pmesh:
        pcm = ax.pcolormesh(pv.lon,pv.lat,pv,transform=mdict['noProj'],cmap=cmap_diff,vmin=vlm[0],vmax=vlm[1])
    else:
        pcm = ax.contourf(pv.lon,pv.lat,pv,transform=mdict['noProj'],cmap=cmap_diff,levels=vlvls)
    cb = fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.05,pad=0.01)
    cb.set_label(title)
    
    
    # Plot contours    
    current = dscurr[1].mean('mon').SSH
    cl = ax.contour(long,latg,current,colors="k",
                    linewidths=0.35,transform=mdict['noProj'],levels=slvls,alpha=0.8)
    ax.clabel(cl)
    
    # Plot Mask
    cl2 = ax.contour(ds_mask.lon,ds_mask.lat,plotmask,colors="w",linestyles='dashed',linewidths=.95,
                    levels=[0,1],transform=mdict['noProj'],zorder=1)

savename = "%s%s_Overall_Variance_Differences.png" % (figpath,expname,)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Zonal and Meridional Averages

#%% Examine the seasonal variances

fsz_title = 20
fsz_axis  = 18
cmap      = 'cmo.thermal'
plot_s    = [0,2,1,3]
fig,axs,mdict = viz.init_orthomap(2,4,bbplot,figsize=(16,7))

slvls = np.arange(-150,160,15)

for ee in range(2):
    
    for s in range(4):
        
        sid = plot_s[s]
        
        ax   = axs[ee,s]
        
        ax   = viz.add_coast_grid(ax,bbox=bbplot,fill_color="k")
        pv   = dsvar_seasonal[ee].mean('ens').isel(season=sid) * msk
        
        if pmesh:
            pcm  = ax.pcolormesh(pv.lon,pv.lat,pv,transform=mdict['noProj'],vmin=0,vmax=vmax,cmap=cmap)
        else:
            pcm  = ax.contourf(pv.lon,pv.lat,pv,transform=mdict['noProj'],levels=levels,extend="both",cmap=cmap)
        
        
        current = dscurr_savg[1].isel(season=sid).SSH
        
        cl = ax.contour(long,latg,current,colors="k",
                        linewidths=0.35,transform=mdict['noProj'],levels=slvls,alpha=0.8)
        ax.clabel(cl)
        
        
        
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

# Make an adjustment to exclude points blowing up (move from 65 to 60 N)
bboxes[0][-1] = 60
bboxes[1][-1] = 60

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
    tsmr   = scm.compute_sm_metrics(rsstin,nsmooth=150)
    
    tsm_regs.append(tsmr)
    ssts_reg.append(ravgs)

#%% Make Some Plots (ACFs)
nregs = len(bboxes)

kmonth = 1
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

mons3   = proc.get_monstr()

if varname == "SST":
    vunit = r"\degree C"
    ylims = [0,0.2]
elif varname == "SSS":
    vunit = r"psu"
    ylims = [0,0.005]

#fig,axs = plt.subplots(2,2,constrained_layout=True,figsize=(10,6.5))
fig,axs = viz.init_monplot(2,2,figsize=(10,6.5))

for rr in range(nregs):
    
    ax   = axs.flatten()[rr]
    
    for ii in range(2):
        plotvar = tsm_regs[rr]['monvars'][ii]
        
        if varname == 'SST':
            plotlab = "%s ($\sigma^2$=%.2f $%s^2$)" % (enames[ii],np.var(ssts_reg[rr][ii]),vunit)
        else:
            plotlab = "%s ($\sigma^2$=%.2e $%s^2$)" % (enames[ii],np.var(ssts_reg[rr][ii]),vunit)
        
        ax.plot(mons3,plotvar,label=plotlab,c=cols[ii],ls=lss[ii],marker=mks[ii])
        
    ax.legend()
    
    ax.set_ylim(ylims)
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
        if varname == "SST":
            plotlab = "%s (var=%.3f $%s^2$)" % (enames[ii],np.var(ssts_reg[rr][ii]),vunit)
        else:
            plotlab = "%s (var=%.5f $%s^2$)" % (enames[ii],np.var(ssts_reg[rr][ii]),vunit)
        
        # Plot Spectra
        ax.plot(plotfreq,plotspec,label=plotlab,c=cols[ii],ls='solid',lw=2.5)
        
        
        # Plot Confidence (this was fitted to whole sepctra, need to limit to lower frequencies)
        plotCCs =  tsm_regs[rr]['CCs'][ii] /dtplot
        ax.plot(plotfreq,plotCCs[:,1] ,c=cols[ii],lw=.75,ls='dotted')
        ax.plot(plotfreq,plotCCs[:,0] ,c=cols[ii],lw=.55,ls='solid')
        
        
        
        
        
        
    ax.legend()
    
    ax.set_xticks(xtks,labels=plotyrs)
    ax.set_xlim([xtks[0],xtks[-1]])
    
    ax.set_xlabel("Period")
    ax.set_ylabel("Power ($psu^2$ / cpy)")
    ax.set_title(regions_sel[rr],fontsize=fsz_title)
savename = "%s%s_Regional_Spectra.png" % (figpath,expname)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Look at actual timeseries

irun  = 0
nruns = 5
fig,axs = plt.subplots(4,1,constrained_layout=True,figsize=(16,14))

for rr in range(nregs):
    
    ax   = axs.flatten()[rr]
    #ax.set_xlim([0,100])
    #ax.set_ylim([0,1.2])
    #ax = viz.add_ticks(ax=ax,)
    #ax,_ = viz.init_monplot()
    
    for run in range(nruns):
        plotvar = ssts_reg[rr][0][run,:]
        ax.plot(plotvar,alpha=0.7,label="Run %02i" % (run+1))
        
    ax.plot(ssts_reg[rr][0][:,:].isel(ens=irun),lw=0.75,
            alpha=1,label="Run %i" % (irun+1),c="k")
    if rr == 0:
        ax.legend(ncol=5)
    ax.set_title(regions_sel[rr],fontsize=fsz_title)
    
    
savename = "%s%s_Regional_Timeseries.png" % (figpath,expname)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Section below has been moved/done in to another script. Delete this.


# ---------------------------------------------------
#%% Part (3) Pointwise Autocorrelation
# ---------------------------------------------------

pathout = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
ncout   = "SM_SST_OSM_Tddamp_SST_autocorrelation_thresALL_lag00to60.nc"

dsout   = xr.open_dataset(pathout + ncout)

acfsout = dsout.SST.isel(thres=0)
T2out   = proc.calc_T2(acfsout,axis=-1)


#%% 



#%% scrap/WIP
#%%

# Based on pointwise_autocorrelation from stochmod/analysis
lags        = np.arange(0,61,1)
lagname     = "lag%02ito%02i" % (lags[0],lags[-1]) 
conf        = 0.95
tails       = 2

ds_in_sm    = ds_in[0]

# First step: transpose the dimensions
invar       = ds_in_sm.transpose('lon','lat','ens','time').values



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

    
    
    

