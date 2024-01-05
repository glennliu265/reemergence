#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize Autocorrelation (Depth v Lag)

Visualizes output of calc_ac_depth.py
Top section copied from calc_ac.py (Until "Load Data")

Created on Tue Jun 14 08:47:26 2022

@author: gliu
"""

import sys
import time
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob

#%% Select dataset to postprocess

# Set Machine
# -----------
stormtrack = 0 # Set to True to run on stormtrack, False for local run

# Data Preprocessing
# ------------------
startyr     = 1920
endyr       = 2006


# SPG Test Point
lonf        = -30+360
latf        = 50

# # # SPG Center
# lonf        = -40+360
# latf        = 53

# # Transition Zone
# lonf = -58 + 360 
# latf = 44

# NE Atlantic
# lonf = -23 + 360 
# latf = 60

# Autocorrelation parameters
# --------------------------
lags        = np.arange(0,61)
lagname     = "lag%02ito%02i" % (lags[0],lags[-1]) 
thresholds  = [0,] # Standard Deviations
conf        = 0.95
tails       = 2

mconfig    = "CESM" #"HadISST" #["PIC-FULL","HTR-FULL","PIC_SLAB","HadISST","ERSST"]
thresholds = [0,]
thresname  = "thres" + "to".join(["%i" % i for i in thresholds])
varname    = "TEMP" # ["TS","SSS","SST]

# MLD Data (located in outpath)
mldname = "CESM1_PiC_HMXL_Clim_Stdev.nc" # Made with viz_mldvar.py (stochmod/analysis)

# Plotting Params
# ---------------
colors   = ['b','r','k']
bboxplot = [-80,0,0,60]
bboxlim  = [-80,0,0,65]
#%% Set Paths for Input (need to update to generalize for variable name)

if stormtrack:
    
    # Module Paths
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
    
    # Input Paths 
    datpath = "/stormtrack/data4/share/deep_learning/data_yuchiaol/cesm_le/TEMP/"
    
    # Output Paths
    figpath = "/home/glliu/02_Figures/01_WeeklyMeetings/20220629/"
    outpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/"
    
else:
    # Module Paths
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")

    # Input Paths 
    datpath = ""
    
    # Output Paths
    figpath     = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20240102/'
    outpath     = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/depth_v_lag/'
    
# Import modules
from amv import proc,viz
import scm

proc.makedir(figpath)
#%% Load Data

# Load the data
savename   = "%s%s_Autocorrelation_DepthvLag_lon%i_lat%i_%s.npz" % (outpath,varname,lonf,latf,lagname)
ld         = np.load(savename,allow_pickle=True)
acs        = ld['acs']         # [ens x depth x mon x thres x lag]
cfs        = ld['cfs']         # [ "                           " x confs ]
counts     = ld['class_count'] # [ens x depth x mon x thres]
thresholds = ld['thresholds']  
lon        = ld['lon']
lat        = ld['lat']
lags       = ld['lags']
threslabs  = ld['threslabs']
z          = ld["z_t"]/100 # Uncomment after script has been rerun




# Flip month and depth order (for rewritten script)
if acs.shape[1] is not len(z):
    acs    = acs.transpose(0,2,1,3,4)
    counts = counts.transpose(0,2,1,3)
    cfs = cfs.transpose(0,2,1,3,4,5)

if varname == "SALT":
    threslabs = [t.replace("T","S") for t in threslabs]
# Temp Fix (will add z to the file above)
#savename2 = "%sCESM1LE_UOTEMP_lon%i_lat%i.nc" % (outpath,lonf,latf)
#ds        = xr.open_dataset(savename2)
#z         = ds.z_t.values/100 # Convert to cm
#ds.close()

#%% Load MLD 
# Get Mean Climatological Cycle
# Get Stdev

dsmld = xr.open_dataset(outpath+mldname)

if lonf > 180:
    lonfw = lonf-360
else:
    lonfw = lonf
    
# Get Mean and Stdev, take maximum of that
hbar     = dsmld.clim_mean.sel(lon=lonfw,lat=latf,method='nearest').values
hstd     = dsmld.stdev.sel(lon=lonfw,lat=latf,method='nearest').values
dsmld.close()


#%% Recalculate confidence level
# Compute 1-sided t-test significance
lagconf = np.zeros(counts.shape)
rhocrit = proc.ttest_rho(conf,1,counts)

# Get dimensions (for ticking, etc)
nens,nz,nmon,nthres,nlags = acs.shape

# Get months for plotting
mons3        = proc.get_monstr(nletters=3)
locfn,locstr = proc.make_locstring(lonf,latf)




#%% Plot I: Depth v. Lag for a single ensemble (or ens avg)

# Plot I, User Edits)
kmonth     = 1
e          = 0#'avg' #6avg' #0  # "avg" # Set the ensemble. Can select "avg"
th         = 2 #"diff" # Threshold to Plot # set "diff" for plotting (Warm - Col)
clvl       = np.arange(-1,1.05,0.05)#np.arange(-1,1.1,.1) #
plotz      = -1 # Index of Maximum Depth Level to Plot
plotlag    = 36 # Index of Maximum Lag to plot
usecontour = True

for e in range(42):
    # Make the mask
    if isinstance(e,int):
        mask    =   (acs[e,...] >= lagconf[e,...,None])
    
    # Tile MLD and MLDvar
    loophbar   = proc.tilebylag(kmonth,hbar,lags)
    loophstd   = proc.tilebylag(kmonth,hstd,lags)
    
    # Prepare Labels and Ticks
    xtk2    = np.arange(0,nlags,3)
    monlabs = viz.prep_monlag_labels(kmonth,xtk2,2)
    
    # Check which threshold to plot
    if th == "diff":
        plotac = acs[:,:plotz,kmonth,1,:plotlag] - acs[:,:plotz,kmonth,0,:plotlag]
        thlab  = "(%s) - (%s)" % (threslabs[1],threslabs[0])
    else:
        plotac = acs[:,:plotz,kmonth,th,:plotlag]
        thlab  = threslabs[th]
    
    # Select the plot
    if isinstance(e,int):
        plotac = plotac[e,:,:]
    else:
        plotac = plotac.mean(0)
    
    # Make the plot
    fig,ax = plt.subplots(1,1,figsize=(14,4))
    if usecontour:
        cf = ax.contourf(lags[:plotlag],z[:plotz],plotac,levels=clvl,cmap='cmo.balance')
    else:
        cf = plt.pcolormesh(lags[:plotlag],z[:plotz],plotac,vmin=clvl[0],vmax=clvl[-1],cmap='cmo.balance',shading='nearest')
    cl = ax.contour(lags[:plotlag],z[:plotz],plotac,levels=clvl,colors='k',linewidths=0.5)
    ax.clabel(cl,fontsize=8)
    
    # Plot Mask
    if isinstance(e,int):
        viz.plot_mask(lags[:plotlag],z[:plotz],mask[:plotz,kmonth,th,:plotlag].T,reverse=True,color='k',markersize=0.5)
    
    # Plot MLD
    ax.plot(lags[:plotlag],loophbar[:plotlag],ls='solid',color="k",lw=1.5)
    ax.plot(lags[:plotlag],loophbar[:plotlag]+loophstd[:plotlag],ls='dotted',color="k",lw=1.5)
    ax.plot(lags[:plotlag],loophbar[:plotlag]-loophstd[:plotlag],ls='dotted',color="k",lw=1.5)
    
    # Axis + Colorbar labeling and formatting
    cb = fig.colorbar(cf,ax=ax)
    cb.set_label("Correlation")
    ax.set_ylim([z[0],z[plotz]])
    plt.gca().invert_yaxis()
    ax.set_xlabel("Lag (Months) from %s" % mons3[kmonth])
    ax.set_xticks(xtk2,)
    ax.set_xticklabels(monlabs)
    ax.set_ylabel("Depth (meters)")
    ax.set_xlim([0,lags[plotlag]])
    
    ax.grid(True,ls='dotted')
    
    ax.set_title("%s Anomaly Lagged Correlation @ %s\n" % (varname,locstr) +
                 "Lag 0: %s, Max Depth: %i m, Ens: %s, Threshold: %s" % (mons3[kmonth],z[plotz],str(e),thlab))
    
    figname = "%sDepthvLag_AC_%s_%s_mon%02i_thres%s_ens%s_lag%i_z%i.png"%(figpath,varname,locfn,kmonth+1,str(th),str(e),lags[plotlag],z[plotz])
    plt.savefig(figname,dpi=150,bbox_inches='tight')
    
    plt.show()

#plt.savefig("%s2D_Reemergence_50N30W.png"%(figpath),dpi=200,bbox_inches='tight')

#%% Plot 2: Check values at a given depth (for all ensemble members)

plotz  = 0

xtk2 = np.arange(0,lags[-1]+3,3)
monlab = viz.prep_monlag_labels(kmonth,xtk2,2)

fig,ax = plt.subplots(1,1,figsize=(8,4))
for th in range(3):
    for e in range(nens):
        ax.plot(lags,acs[e,plotz,kmonth,th,:],
                color=colors[th],alpha=0.05,label="")
    ax.plot(lags,acs[:,plotz,kmonth,th,:].mean(0),
            color=colors[th],label=threslabs[th])
    
ax.legend()
ax.set_xlim([lags[0],lags[-1]])
ax.set_title("%s Autocorrelation @ %s, z=%i m"  % (mons3[kmonth],locstr,z[plotz]))
#ax.set_ylim([-0.1,1])
ax.axhline(0,ls='dashed',lw=0.5,color="k")
ax.grid(True,ls='dotted')
ax.set_xticks(xtk2,monlab)

figname = "%sAC_%s_%s_mon%02i_thres%i_z%i.png"%(figpath,varname,locfn,kmonth+1,th,z[plotz])
plt.savefig(figname,dpi=150,bbox_inches='tight')

plt.show()

#%% Plot 3: For given threshold and month, plot for all ensemble members

th      = 2
kmonth  = 1

plotz   = -1 
plotlag = 36

# Tile MLD and MLDvar
loophbar   = proc.tilebylag(kmonth,hbar,lags)
loophstd = proc.tilebylag(kmonth,hstd,lags)

fig,axs = plt.subplots(6,7,figsize=(24,14),constrained_layout=True)
for e in tqdm(range(42)):
    ax = axs.flatten()[e]
    ax = viz.label_sp("ens%02i"% (e+1),ax=ax,labelstyle="%s",usenumber=True,alpha=0.8)

    
    plotac = acs[e,:plotz,kmonth,th,:plotlag]
    mask   = (acs[e,...] >= lagconf[e,...,None])
    
    # Plot Contours
    cf     = ax.contourf(lags[:plotlag],z[:plotz],plotac,levels=clvl,cmap='cmo.balance')
    cl = ax.contour(lags[:plotlag],z[:plotz],plotac,levels=clvl,colors='k',linewidths=0.25)
    ax.clabel(cl,clvl[:2],fontsize=8)
    
    # Plot Mask
    viz.plot_mask(lags[:plotlag],z[:plotz],mask[:plotz,kmonth,th,:plotlag].T,ax=ax,reverse=False,color='k',markersize=0.5)
    
    
    # Plot MLD
    ax.plot(lags[:plotlag],loophbar[:plotlag],ls='solid',color="k",lw=0.75)
    ax.plot(lags[:plotlag],loophbar[:plotlag]+loophstd[:plotlag],ls='dotted',color="k",lw=.75)
    ax.plot(lags[:plotlag],loophbar[:plotlag]-loophstd[:plotlag],ls='dotted',color="k",lw=.75)
    
    # Axis + Colorbar labeling and formatting
    ax.set_ylim([z[0],z[plotz]])
    ax.invert_yaxis()
    ax.set_xticks(xtk2)
    ax.set_xlim([0,lags[plotlag]])
    ax.grid(True,ls='dotted')

plt.suptitle("%s %s AC @ %s, Thres: %s" % (mons3[kmonth],varname,locstr,threslabs[th]))
figname = "%sDepthvLag_AC_%s_%s_mon%02i_thres%i_ALLens_lag%i_z%i.png"%(figpath,varname,locfn,kmonth+1,th,lags[plotlag],z[plotz])
plt.savefig(figname,dpi=150,bbox_inches='tight')
plt.show()

#%% Making a new re-emergence index

def calc_remidx_simple(ac,kmonth,monthdim=-2,lagdim=-1,
                       minref=6,maxref=12,tolerance=3,debug=False):
    
    # Select appropriate month
    ac_in          = np.take(ac,np.array([kmonth,]),axis=monthdim).squeeze()
    
    # Compute the number of years involved (month+lags)/12
    nyrs           = int(np.floor((ac_in.shape[-1] + kmonth) /12))
    
    # Move lagdim to the front
    ac_in,neworder = proc.dim2front(ac_in,lagdim,return_neworder=True)
    
    # Make an array
    #remidx     = np.zeros((nyrs,),ac_in.shape[1:])    # [year x otherdims]
    maxmincorr = np.zeros((2,nyrs,)+ac_in.shape[1:])  # [max/min x year x otherdims]
    
    for yr in range(nyrs):
        
        # Get indices of target lags
        minid = np.arange(minref-tolerance,minref+tolerance+1,1) + (yr*12)
        maxid = np.arange(maxref-tolerance,maxref+tolerance+1,1) + (yr*12)
        
        # Drop indices greater than max lag
        maxlag = (ac_in.shape[0]-1)
        minid  = minid[minid<=maxlag]
        maxid  = maxid[maxid<=maxlag]
        
        if debug:
            print("For yr %i"% yr)
            print("\t Min Ids are %i to %i" % (minid[0],minid[-1]))
            print("\t Max Ids are %i to %i" % (maxid[0],maxid[-1]))
        
        # Find minimum
        mincorr  = np.min(np.take(ac_in,minid,axis=0))
        maxcorr  = np.max(np.take(ac_in,maxid,axis=0))
        
        maxmincorr[0,yr,...] = mincorr.copy()
        maxmincorr[1,yr,...] = maxcorr.copy()
        #remreidx[yr,...]     = (maxcorr - mincorr).copy()
    
    return maxmincorr



maxmincorr = calc_remidx_simple(acs,1,monthdim=2,lagdim=-1,debug=True)
remidx     = maxmincorr[1,...] - maxmincorr[0,...]
#%% More complicated version, under construction...


def calc_remidx(ac,kmonth,monthdim=-2,lagdim=-1,
                min_mons=np.arange(2,8),
                max_mons=np.hstack([np.arange(8,12),np.arange(0,3)])):
                summer_base=6,
                winter_base=12):
    """
    Calculate (wintertime) re-emergence strength for autocorrelation [...,kmonth,lag].
    
    By default, looks for:
        - A minima from Mar - Sept
        - A maxima from Sept - Feb
    """
    
    # Select appropriate month
    ac_in = np.take(ac,np.array([kmonth,]),axis=monthdim).squeeze()
    
    # Compute the number of years involved (month+lags)/12
    nyrs = np.floor((ac_in.shape[-1] + kmonth) /12)
    
    # Looping for each year
    maxcorr = []
    mincorr = []
    remidx  = []
    for yr in range(nyrs):
        
        idx_min = min_mons - kmonth + yr
        
        # Get indices of months
        idx_max = max_mons - kmonth + yr
        idx_max = [i+12 if i<=kmonth else i for i in idx_max]
        
        
        
    # Adjust search months depending on base month
    
    # Looping for each month Find max/min, take diff in correlation
    
    # First, based on starting month, compute the window
    
    

    
    
    
    
