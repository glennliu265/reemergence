#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize Metrics computed by [postprocess_smoutput_auto]

Copied segments over from postprocess_smoutput
Note: currently works for Astraeus

Created on Wed Mar 13 11:11:00 2024

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
import cartopy.crs as ccrs

# ----------------------------------
#%% Import custom modules and paths
# ----------------------------------

# Indicate the Machine!
machine = "Astraeus"

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
import stochmod_params as sparams

# Set needed paths
figpath     = pathdict['figpath']
input_path  = pathdict['input_path']
output_path = pathdict['output_path']
procpath    = pathdict['procpath']

# Make Needed Paths
proc.makedir(figpath)

#%% Indicate Experiment Information

expname      = "SSS_EOF_Qek_LbddEnsMean"
varname      = "SSS"

#%% Set Metrics Path



metrics_path = output_path + expname + "/Metrics/" 



print("Performing Postprocessing for %s" % expname)
print("\tSearching for Metrics in %s" % metrics_path)

# -----------------------
#%% Load Computed Metrics
# -----------------------

# Load Overall Variance, dims  : (run: 10, lat: 48, lon: 65)
savenamevar    = "%sPointwise_Variance.nc" % (metrics_path)
dsvar_byens    = xr.open_dataset(savenamevar).load()

# Load Seasonal Variance, dims : (run: 10, season: 4, lat: 48, lon: 65)
savenamevar    = "%sPointwise_Variance_Seasonal.nc" % (metrics_path) 
dsvar_seasonal = xr.open_dataset(savenamevar).load()


#%% Load Information for visualization


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



# ---------------------------------------------------------------------
#%% Load CESM1 Output for SSS (copied from analyze_basinwide_output_SSS)
# ---------------------------------------------------------------------
# Need to update this at some point

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

# -----------------------------
#%% SET PLOTTING PARAMETERS
# -----------------------------

ds                          = dsvar_byens
lon                         = ds.lon.values
lat                         = ds.lat.values
#tsim           = ds.time.values

latf                        = 50
lonf                        = -30
klon,klat                   = proc.find_latlon(lonf,latf,lon,lat)
locfn,loctitle              = proc.make_locstring(lonf,latf)

# Other Plotting Options
bbplot                      = [-80,0,20,65]
mpl.rcParams['font.family'] = 'Avenir'
proj                        = ccrs.PlateCarree()

# FONT SIZES
fsz_title                   = 20

#%% Load Land Ice Mask (Copied from visualize_acf_rmse)

# Set names for land ice mask (this is manual, and works just on Astraeus :(...!)
lipath           = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/Masks/"
liname           = "CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc"

# Load Land Ice Mask
ds_mask          = xr.open_dataset(lipath+liname).MASK.squeeze().load()

# Edit
plotmask = ds_mask.values.copy()
plotmask[np.isnan(plotmask)] = 0.

maskcoast = ds_mask.values.copy()
maskcoast = np.roll(maskcoast,1,axis=0) * np.roll(maskcoast,-1,axis=0) * np.roll(maskcoast,1,axis=1) * np.roll(maskcoast,-1,axis=1)

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



# ---------------------------------------------------
#%% Part (1) Overall Variance
# ---------------------------------------------------

dsvar_byens    = [ds.std('time')  for ds in ds_in]
dsvar_seasonal = [ds.groupby('time.season').std('time') for ds in ds_in]




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


