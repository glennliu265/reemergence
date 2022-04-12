#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Experiment with AR1 and Noise timeseries to see how things vary
Created on Tue Apr 12 11:15:00 2022
@author: gliu
"""

import numpy as np
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")

from amv import proc,viz
import scm
from tqdm import tqdm

from scipy import stats

import matplotlib.pyplot as plt
#%%

datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/"
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20220414/"
proc.makedir(figpath)
sstpts  = scm.load_cesm_pt(datpath,grabpoint=[-30,50])
ths     = [0,]
colors  = ["b","r"]

lags    = np.arange(0,37,1)

sstpt.shape
debug = True

#%%

def make_ar1(r1,sigma,simlen,t0=0,savenoise=False,usenoise=None):
    """
    Create AR1 timeseries given the lag 1 corr-coef [r1],
    the amplitude of noise (sigma), and simluation length.
    
    Adapted from slutil.return_ar1_model() on 2022.04.12

    Parameters
    ----------
    r1        [FLOAT] : Lag 1 correlation coefficient
    sigma     [FLOAT] : Noise amplitude (will be squared)
    simlen    [INT]   : Simulation Length
    t0        [FLOAT] : Starting value (optional, default is 0.)
    savenoise [BOOL]  : Output noise timeseries as well
    usenoise  [ARRAY] : Use provided noise timeseries

    Returns
    -------
    rednoisemodel : [ARRAY: (simlen,)] : AR1 model
    noisets       : [ARRAY: (simlen,)] : Used noise timeseries
    """
    # Create Noise
    if usenoise is None:
        noisets       = np.random.normal(0,1,simlen)
        noisets       *= (sigma**2) # Scale by sigma^2
    else:
        noisets = usenoise
    # Integrate Model
    rednoisemodel = np.zeros(simlen) * t0 # Multiple by initial value
    for t in range(1,simlen):
        rednoisemodel[t] = r1 * rednoisemodel[t-1] + noisets[t]
    if savenoise:
        return rednoisemodel,noisets
    return rednoisemodel

# -----------------------
#%% Create AR1 Timeseries
# -----------------------

simlen = 12 * 10000
sigmas = np.arange(0.2,2.2,0.2)
r1s    = np.arange(0,1.1,0.1)

# Make the synthetic timeseries
# -----------------------------
nsig   = len(sigmas)
nr1    = len(r1s)
ts_all   = np.zeros((nr1,nsig,simlen,)) * np.nan # [r1 x sigma x time]
expnames       = np.zeros((nr1,nsig,),dtype=object) 
expnames_fancy = expnames.copy()
it = 0
for r in tqdm(range(nr1)):
    r1 = r1s[r]
    for s in range(nsig):
        sig = sigmas[s]
        
        # Generate timeseries
        if it == 0: # Save the noise
            ts_all[r,s,:],noisets = make_ar1(r1,sig,simlen,t0=0,savenoise=True)
            noisets /= (sig**2) # Divide back to get unity
        else: # 
            ts_all[r,s,:] = make_ar1(r1,sig,simlen,t0=0,usenoise=noisets)
            
        # Make string names for plotting
        expnames[r,s]       = "r%.02f_sigma%.02f" % (r1,sig)
        expnames_fancy[r,s] = r"$\rho_1$: %.02f, $\sigma$: %.02f" % (r1,sig)
        
# ------------------------------
#%% Compute +/- autocorrelations
# ------------------------------
im       = 0 # Select an arbitrary month
cid      = 0 # Class id. 0 = neg, 1 = pos, 2 = all
spearman = False

# All this will be looped
s     = 1
r     = 9


# Preallocate and loop
nlag    = len(lags)
nthres  = len(ths) + 2
t_lags  = np.zeros((nr1,nsig,nthres),dtype='object')
t_bases = t_lags.copy()
acs     = np.zeros((nr1,nsig,nthres,nlag)) * np.nan
        
for s in tqdm(range(nsig)):
    for r in range(nr1):
        ts_in = ts_all[r,s,:]
        
        
        # Prepare the input and reshape 
        nmon    = ts_in.shape[0]
        nyr     = int(nmon/12)
        sstpt   = ts_in.reshape(nyr,12) # [year x mon]
        #print("For s %i, r %i: sst shape is %s" % (s,r,sstpt.shape))
        
        # Select a month
        sst_mon       = sstpt[:,im]
        sstpt_classes = proc.make_classes_nd(sst_mon[:,None],ths,dim=0,debug=False)
        

        
        # Compute the autoorrelation
        for th in range(len(ths)+2):
            
            # Get the yr_mask
            yr_mask  = np.where(sstpt_classes==th)[0]
            if th == 2:
                yr_mask = np.arange(0,sst_mon.shape[0])
                
            # Calculate lagged autocorrelation
            ac,t_base,t_lag = proc.calc_lagcovar(sstpt.T,sstpt.T,lags,im+1,0,
                                                 yr_mask=yr_mask,debug=False,return_values=True,
                                                 spearman=spearman)
            
            # Save to output
            t_bases[r,s,th] = t_base # [r1 x sig x th]
            t_lags[r,s,th]  = t_lag  # [r1 x sig x th][lag]
            acs[r,s,th,:]      = ac     # [r1 x sig x lag]

#%% Make scatterplots of reference vs lag month for selected lags
r = 0
s = -4

# Select base and lagged variable
baseNone  = t_bases[r,s,-1]
lagsNone  = t_lags[r,s,-1]
basemask  = t_bases[r,s,0]
lagsmask  = t_lags[r,s,0]

# Select which lags to plot
plotlags  = np.arange(1,9)
plotlags  = [0,1,3,5,10,15,25,35]

# Set Plot limits and initialize
lms = 10
fig,axs = plt.subplots(2,4,figsize=(26,10))

for i,ilag in tqdm(enumerate(plotlags)):
    
    #ilag = plotlags[i] #i+1 # Start from 1
    ax = axs.flatten()[i]
    ax.set_aspect('equal','box')
    
    ax.scatter(baseNone,lagsNone[ilag],alpha=0.5,color='blue',marker="d",label="ALL")
    ax.scatter(basemask,lagsmask[ilag],alpha=0.5,color='yellow',marker="d",label="MASKED")
    
    ax.plot([0, 1], [0, 1], transform=ax.transAxes,color='k')
    ax.grid(True,ls='dotted')
    
    ax.set_title(r"Lag %i, $\rho_{ALL}$=%.2f, $\rho_{neg}$=%.2f" % (ilag,acs[r,s,-1,ilag],acs[r,s,0,ilag]))
    ax.set_xlim([-lms,lms])
    ax.set_ylim([-lms,lms])
i = 0
plt.suptitle(expnames_fancy[r,s])
plt.savefig("%sAutocorrelation_Scatter_subset_%s.png"% (figpath,expnames[r,s]),dpi=200)

#%% Plot parameter matrix, as suggested by Chris

ilag   = 1

vmin   = -.2
vmax   = .2

fig,ax = plt.subplots(1,1,figsize=(6,6))
pcm = ax.pcolormesh(sigmas,r1s,acs[:,:,0,ilag] - acs[:,:,-1,ilag],
                    cmap='cmo.balance',shading='nearest',vmin=vmin,vmax=vmax)
ax.set_aspect('equal','box')

cb = fig.colorbar(pcm,ax=ax,orientation='vertical',fraction=0.025,pad=0.01)
cb.set_label(r"$\rho_{ALL}$ - $\rho_{NEGATIVE}$")
#ax.grid(True,ls='solid',color="w")
ax.set_xticks(sigmas)
ax.set_yticks(r1s)

ax.set_title("Difference in Pearson's R for Lag %i"% (ilag))

ax.set_ylabel(r"$\rho_1$")
ax.set_xlabel(r"$\sigma^2$")

#%% Visualize the lag correlation


r = 4
s = 4

labels = ["(-) Anomalies","(+) Anomalies","All Anomalies"]
colors = ["b","r","k"]

fig,ax = plt.subplots(1,1)
title  = "Autocorrelation for Simulation: %s" % expnames_fancy[r,s]
ax,ax2     = viz.init_acplot(im,np.arange(0,36+3,3),lags,ax=ax,title=title)
for th in range(3):
    ax.plot(lags,acs[r,s,th,:],label=labels[th],c=colors[th])
ax.legend()

#%%


#%% Visualize/Debug results


#%%


sstpt   = sstpts[0]
nmon    = sstpt.shape[0]
nyr     = int(nmon/12)
sstpt   = sstpt.reshape(nyr,12) # [year x mon]

im      = 1
#%%

sst_mon       = sstpt[:,im]
sstpt_classes = proc.make_classes_nd(sst_mon[:,None],ths,dim=0,debug=True)
#%%
t = np.arange(0,nmon,1)
tyr = t.reshape(nyr,12)

xyr = np.arange(0,nyr,1)
fig,ax = plt.subplots(1,1)

#ax.plot(sstpt.flatten())
ax.plot(xyr,sst_mon)
ax.set_xlim([0,100])

for n,th in enumerate(range(len(ths)+1)):
    if th < len(ths):
        ax.axhline(sstpt.std()*ths[th],ls='dashed',color='k')
    
    id_sel = np.where(sstpt_classes == th)[0]
    ax.scatter(xyr[id_sel],sst_mon[id_sel],marker="x")
    
    
    
    
    #ax.scatter(tyr[id_sel,:].flatten(),sstpt[id_sel,:].flatten(),color=colors[n],marker="x")


    
ax.set_xlabel("Year")


# TASK
# Modify function to output SSTs considered at each lag (or just visualize it)
    
#%%
# im = 1
# pt = 3333
# th = 1
#yr_mask     = np.where(sst_mon_classes[pt,:] == th)[0] # Indices of valid years
#testvar     = sst_valid[pt,:,:] # [pts,nyr,nmon] # [yr, sst]

testvar  = sstpt

yr_mask  = np.where(sstpt_classes==1)[0]

# Input Arguments
var1 = testvar.T # mon x year
var2 = testvar.T
lags = np.arange(0,37,1)
basemonth = im+1
detrendopt = 0
debug = True

def calc_lagcovar(var1,var2,lags,basemonth,detrendopt,yr_mask=None,debug=True,
                  return_values=False):
    """
    Calculate lag-lead relationship between two monthly time series with the
    form [mon x yr]. Lag 0 is set by basemonth
    
    Correlation will be calculated for each lag in lags (lead indicate by
    negative lags)
    
    Set detrendopt to 1 for a linear detrend of each time series.
    
    
    Inputs:
        1) var1: Monthly timeseries for variable 1 [mon x year]
        2) var2: Monthly timeseries for variable 2 [mon x year]
        3) lags: lags and leads to include
        4) basemonth: lag 0 month
        5) detrendopt: 1 for linear detrend of both variables
        6) yr_mask : ARRAY of indices for selected years
        7) debug : Print check messages
    
    Outputs:
        1) corr_ts: lag-lead correlation values of size [lags]
        2) yr_count : print the count of years
        3) varbase : [yrs] Values of monthly anomalies for reference month
        4) varlags : [lag][yrs] Monthly anomalies for each lag month
    
    Dependencies:
        numpy as np
        scipy signal,stats
    
    """
    
    # Get total number of lags
    lagdim = len(lags)
    
    # Get timeseries length
    totyr = var1.shape[1]
    
    # Get total number of year crossings from lag
    endmonth = basemonth + lagdim-1
    nlagyr   = int(np.ceil(endmonth/12)) #  Ignore zero lag (-1)
    
    if debug:
        print("Lags spans %i mon (%i yrs) starting from mon %i" % (endmonth,nlagyr,basemonth))
        
    # Get Indices for each year
    if yr_mask is not None:
        # Drop any indices that are larger than the limit
        # nlagyr-1 accounts for the base year...
        # totyr-1 accounts for indexing
        yr_mask_clean = np.array([yr for yr in yr_mask if (yr+nlagyr-1) < totyr])
        
        if debug:
            n_drop = np.setdiff1d(yr_mask,yr_mask_clean)
            print("Dropped the following years: %s" % str(n_drop))
        
        yr_ids  = [] # Indices to 
        for yr in range(nlagyr):
            
            # Apply year-lag to index
            yr_ids.append(yr_mask_clean + yr)
    
    
    # Get lag and lead sizes (in years)
    leadsize = int(np.ceil(len(np.where(lags < 0)[0])/12))
    lagsize = int(np.ceil(len(np.where(lags > 0)[0])/12))
    
    # Detrend variables if option is set
    if detrendopt == 1:
        var1 = signal.detrend(var1,1,type='linear')
        var2 = signal.detrend(var2,1,type='linear')
    
    # Get base timeseries to perform the autocorrelation on
    if yr_mask is not None:
        varbase = var1[basemonth-1,yr_ids[0]] # Anomalies from starting year
    else: # Use old indexing approach
        base_ts = np.arange(0+leadsize,totyr-lagsize)
        varbase = var1[basemonth-1,base_ts]
        
    # Preallocate Variable to store correlations
    corr_ts = np.zeros(lagdim)
    
    # Set some counters
    nxtyr = 0
    addyr = 0
    modswitch = 0
    
    varlags = [] # Save for returning later
    for i in lags:

        lagm = (basemonth + i)%12
        
        if lagm == 0:
            lagm = 12
            addyr = 1         # Flag to add to nxtyr
            modswitch = i+1   # Add year on lag = modswitch
            
        if addyr == 1 and i == modswitch:
            if debug:
                print('adding year on '+ str(i))
            addyr = 0         # Reset counter
            nxtyr = nxtyr + 1 # Shift window forward
            
        # Index the other variable
        if yr_mask is not None:
            varlag = var2[lagm-1,yr_ids[nxtyr]]
            if debug:
                print("For lag %i (m=%i), first (last) indexed year is %i (%i) " % (i,lagm,yr_ids[nxtyr][0],yr_ids[nxtyr][-1]))
        else:
            lag_ts = np.arange(0+nxtyr,len(varbase)+nxtyr)
            varlag = var2[lagm-1,lag_ts]
            if debug:
                print("For lag %i (m=%i), lag_ts is between %i and %i" % (i,lagm,lag_ts[0],lag_ts[-1]))
            
        varbase = varbase - varbase.mean()
        varlag  = varlag - varlag.mean()
        #print("Lag %i Mean is %i ")
        
        # Calculate correlation
        corr_ts[i] = stats.pearsonr(varbase,varlag)[0]
        varlags.append(varlag)
        
    if return_values:
        return corr_ts,varbase,varlags
    if yr_mask is not None:
        return corr_ts,len(yr_ids[-1]) # Return count of years as well
    return corr_ts


# Plot a result
test_None,baseNone,lagsNone = calc_lagcovar(var1,var2,lags,basemonth,detrendopt,yr_mask=None,debug=True,return_values=True)
test_mask,basemask,lagsmask = calc_lagcovar(var1,var2,lags,basemonth,detrendopt,yr_mask=yr_mask,debug=True,return_values=True)


fig,ax = plt.subplots(1,1)
ax.plot(lags,test_None,label="No MASK (%i)"%len(sstpt))
ax.plot(lags,test_mask,label="With MASK (%i)"%len(yr_mask))
ax.legend()
#%% What if I run a non-parametric approach (sample with replacement)

nmc         = 1000 # Number of MCS
sample_size = 50 # Number of indices to select randomly
shuffids = []
acs_mcs  = np.zeros((len(test_mask),nmc)) # Lag x Iteration

for i in tqdm(range(nmc)):
    sample_ids = np.random.choice(yr_mask,size=sample_size)
    shuffids.append(sample_ids)
    test = calc_lagcovar(var1,var2,lags,basemonth,detrendopt,yr_mask=sample_ids,debug=False)
    
    acs_mcs[:,i] = test[0].copy()

#%% Compare MCS simulations and original results
    
fig,ax = plt.subplots(1,1)


for i in range(nmc):
    ax.plot(lags,acs_mcs[:,i],alpha=0.01,color="k")
    
ax.plot(lags,test_None,label="No MASK (%i)"%len(sstpt))
ax.plot(lags,test_mask,label="With MASK (%i)"%len(yr_mask))

ax.plot(lags,acs_mcs.mean(1),label="MCS Mean",color='k')

ax.legend()

plt.savefig("%sMCS_Attempt_Autocorrelation.png"%figpath,dpi=150)

#%% Plot the covariance plots

# fig,axs = plt.subplots(3,12,figsize=(20,6))


# for i in range(36):
    
#     ilag = i+1 # Start from 1
#     ax = axs.flatten()[i]
#     ax.set_aspect('equal','box')
#     ax.scatter(baseNone,lagsNone[ilag])
#i = 0

ilag =0
fig,ax = plt.subplots(1,1,figsize=(6,6))

ax.set_aspect('equal','box')

ax.scatter(baseNone,lagsNone[ilag],alpha=0.5,color='blue',marker="d",label="ALL (%i)"%(len(sstpt)))
ax.scatter(basemask,lagsmask[ilag],alpha=0.5,color='yellow',marker="d",label="MASKED"%(len(yr_mask)))

ax.plot([0, 1], [0, 1], transform=ax.transAxes,color='k')
ax.grid(True,ls='dotted')

ax.set_xlabel("Reference Month SST (K)")
ax.set_ylabel("Lag Month SST (K)")
ax.set_title("Lag = %i Months \n Correlation: ALL (%.3f) | MASKED (%.3f)" % (ilag,test_None[ilag],test_mask[ilag]))


ax.legend()

#%%

lms = 4
fig,axs = plt.subplots(3,12,figsize=(26,10))

for i in tqdm(range(36)):
    
    ilag = i+1 # Start from 1
    ax = axs.flatten()[i]
    ax.set_aspect('equal','box')
    
    ax.scatter(baseNone,lagsNone[ilag],alpha=0.5,color='blue',marker="d",label="ALL")
    ax.scatter(basemask,lagsmask[ilag],alpha=0.5,color='yellow',marker="d",label="MASKED")
    
    ax.plot([0, 1], [0, 1], transform=ax.transAxes,color='k')
    ax.grid(True,ls='dotted')
    
    ax.set_title("Lag %i" % ilag)
    ax.set_xlim([-lms,lms])
    ax.set_ylim([-lms,lms])

i = 0


plt.savefig("%sAutocorrelation_Scatter.png"%figpath,dpi=200)

#%% Plot a subset of lags

lms = 4
fig,axs = plt.subplots(2,4,figsize=(26,10))

for i in tqdm(range(8)):
    
    ilag = i+1 # Start from 1
    ax = axs.flatten()[i]
    ax.set_aspect('equal','box')
    
    ax.scatter(baseNone,lagsNone[ilag],alpha=0.5,color='blue',marker="d",label="ALL")
    ax.scatter(basemask,lagsmask[ilag],alpha=0.5,color='yellow',marker="d",label="MASKED")
    
    ax.plot([0, 1], [0, 1], transform=ax.transAxes,color='k')
    ax.grid(True,ls='dotted')
    
    ax.set_title("Lag %i" % ilag)
    ax.set_xlim([-lms,lms])
    ax.set_ylim([-lms,lms])

i = 0
plt.suptitle("")
plt.savefig("%sAutocorrelation_Scatter(subset.png"%figpath,dpi=200)
#%%

test0,cnt0 = proc.calc_lagcovar(var1,var2,lags,basemonth,detrendopt,yr_mask=yr_mask,debug=True)
test_None  = proc.calc_lagcovar(var1,var2,lags,basemonth,detrendopt,yr_mask=None,debug=True)

test_all,cnt_all = proc.calc_lagcovar(var1,var2,lags,basemonth,detrendopt,yr_mask=np.arange(0,var1.shape[-1]),debug=True)


#%%
fig,ax = plt.subplots(1,1)
test0,cnt0 = proc.calc_lagcovar(var1,var2,lags,basemonth,detrendopt,yr_mask=yr_mask,debug=True)

ax.plot(lags,test0,label="Original MASK")

#ax.plot(lags,test_None,label="NO MASK")
#ax.plot(lags,test_all,label="ALL MASK")

ax.legend()

#%% Plot ACF where different numbers of points are skipped

fig,ax = plt.subplots(1,1)

for i in range(20):
    #testp,cnt_all = proc.calc_lagcovar(var1,var2,lags,basemonth,detrendopt,yr_mask=np.arange(0,var1.shape[-1]),debug=True)
    testp,cnt_all = proc.calc_lagcovar(var1,var2,lags,basemonth,detrendopt,yr_mask=yr_mask[i:],debug=True)

    ax.plot(lags,testp,label="%i-skip MASK"%i)

#ax.plot(lags,test_None,label="NO MASK")
#ax.plot(lags,test_all,label="ALL MASK")

ax.legend()

#%% Quantifying decay by fitting an exponential function

lagarray = np.array(lagsmask)
plt.plot(lags,lagarray-lagarray[0],alpha=0.2)

