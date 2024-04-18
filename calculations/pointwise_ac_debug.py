#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute pointwise crosscorrelation for 2 variables
For a given model output.

[Copied from pointwise_autocorrelation_lens.py]

Support separate calculation for positive and negative anomalies, based on the base variable.

Based on postprocess_autocorrelation.py

Created on Thu Mar 17 17:09:18 2022
@author: gliu
"""

import sys
import time
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm

import matplotlib as mpl
mpl.rcParams['font.family'] = 'JetBrains Mono'

from scipy import stats

#%% User Edits

stormtrack = False

# Autocorrelation parameters
# --------------------------
lags        = np.arange(0,61)
lagname     = "lag%02ito%02i" % (lags[0],lags[-1]) 
thresholds  = [-1,1] # Standard Deviations
conf        = 0.95
tails       = 2

# Dataset Parameters
# ---------------------------
outname_data = "CESM1_1920to2005_SSTACF"#"CESM1_1920to2005_SSTvSSS"
vname_base   = "SST"
vname_lag    = "SST"
nc_base      = "CESM1LE_SST_NAtl_19200101_20050101_bilinear.nc" # [ensemble x time x lat x lon 180]
nc_lag       = "CESM1LE_SST_NAtl_19200101_20050101_bilinear.nc" # [ensemble x time x lat x lon 180]
datpath      = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"
preprocess   = True # If True, demean (remove ens mean) and deseason (remove monthly climatology)

# Output Information
# -----------------------------
outpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
figpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/02_Figures/20240322/"

# Mask Loading Information
# ----------------------------
# Set to False to not apply a mask (otherwise specify path to mask)
loadmask    = False #"/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/limask180_FULL-HTR.npy"

# Load another variable to compare thresholds (might need to manually correct)
# ----------------------------------------------------------------------------
# CAUTION: This has not been updated from original script...
thresvar      = False #
thresvar_name = "HMXL"
thresvar_path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/thresholdvar/HMXL_FULL_PIC_lon-80to0_lat0to65_DTNone.nc"
if thresvar is True:
    print("WARNING NOT IMPLEMENTED. See Old Script...")
    # loadvar = xr.open_dataset(thresvar_path)
    # loadvar = loadvar[thresvar_name].values.squeeze() # [ensemble x time x lat x lon]
    
    # # Adjust dimensions to [lon x lat x time x (otherdims)]
    # loadvar = loadvar.transpose(2,1,0)#[...,None]

# Other Information
# ----------------------------
colors   = ['b','r','k']
bboxplot = [-80,0,0,60]
bboxlim  = [-80,0,0,65]
debug    = False

#%% Set Paths for Input (need to update to generalize for variable name)

if stormtrack:
    # Module Paths
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
else:
    # Module Paths
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")

# Import modules
from amv import proc,viz
import scm

# ----------------
#%% Load the data
# ----------------

lonf = -30
latf = 50

# Uses output similar to preprocess_data_byens
# [ens x time x lat x lon]

st     = time.time()
    
# Load Variables
ds_base = xr.open_dataset(datpath+nc_base).sel(lon=lonf,lat=latf,method='nearest').load()
ds_lag  = xr.open_dataset(datpath+nc_lag).sel(lon=lonf,lat=latf,method='nearest').load()

# Make sure they are the same size
#ncs_raw        = [ds_base,ds_lag]
#ncs_resize     = proc.resize_ds(ncs_raw)
#ds_base,ds_lag = ncs_resize

# Get Lat/Lon
lon            = ds_base.lon.values
lat            = ds_base.lat.values
times          = ds_base.time.values
#bbox_base      = proc.get_bbox(ds_base)
print("Loaded data in %.2fs"% (time.time()-st))

# --------------------------------
#%% Apply land/ice mask if needed
# --------------------------------
if loadmask:
    print("Applying mask loaded from %s!"%loadmask)
    
    # Load the mask
    msk  = xr.open_dataset(loadmask) # Lon x Lat (global)
    
    # Restrict to the same region
    dsin  = [ds_base,msk]
    dsout = proc.resize_ds(dsin) 
    _,msk = dsout
    
    # Apply to variables
    ds_base = ds_base * msk
    ds_lag  = ds_lag * msk
    
# -----------------------------
#%% Preprocess, if option is set
# -----------------------------

def preprocess_ds(ds):
    # Remove mean seasonal cycle
    dsa = proc.xrdeseason(ds)
    dsa = dsa - dsa.mean('ensemble')
    return dsa

if preprocess:
    st     = time.time()
    dsin   = [ds_base,ds_lag]
    dsanom = [preprocess_ds(ds) for ds in dsin]
    ds_base,ds_lag = dsanom
    print("Preprocessed data in %.2fs"% (time.time()-st))


#%%
#%% Check the ACF
#%%


# Get Dimensions
varsin               = [ds_base[vname_base].values,ds_lag[vname_lag].values]
nlon,nlat,nens,ntime = varsin[0][None,None,:,:].shape
npts                 = 1
nyrs                 = int(ntime/12)
varsin               = [v.reshape(npts,nens,nyrs,12) for v in varsin] #  (1, 42, 86, 12)

mons3 = proc.get_monstr()

thresnames = ["Negative","Neutral","Positive","ALL"]
threscols  = ["royalblue","gray","orangered","black"]
locfn,_ = proc.make_locstring(lonf,latf,lon360=True)

detrend_anom = True

#%%
# Loop by month
e  = 0
im = 0

nthres     = len(thresholds) + 2
nlags      = len(lags)
acs_all    = np.zeros((nens,12,nthres,nlags))
counts_all = np.zeros((nens,12,nthres))

for e in tqdm(range(nens)):
    
    for im in range(12):
        
        varsin_ens       = [v[:,e,:,:].squeeze() for v in varsin]  # [86,12]
        varsmon          = [v[:,e,:,im].squeeze() for v in varsin] # (86)
        
        data_mon_classes,thresset = proc.make_classes_nd(varsmon[0],thresholds,dim=0,debug=False,return_thres=True)
        data_mon_classes = data_mon_classes.squeeze() # Use the Base Variable
        
        # Make a plot
        if debug:
            
            yrs    = np.arange(1920,2006,1)
            fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(12,4))
            
            ax.plot(yrs,varsmon[0],color='darkred')
            ax.axhline([0],lw=0.5,ls='solid',color="k")
            sigma = np.std(varsmon[0])
            ax.axhline([-sigma],lw=0.5,ls='dotted',color="k")
            ax.axhline([sigma],lw=0.5,ls='dotted',color="k")
            ax.scatter(yrs,varsmon[0],c=data_mon_classes)
            
            classcount = [np.sum(data_mon_classes==c) for c in range(3)]
            ax.set_title("Mon %02i Ens %02i\n Class Counts: %s" % (im+1,e+1,str(classcount)))
            
            
            def init_scatter(figsize=(6,6),vlims=[-2,2]):
                fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=figsize)
                ax     = viz.add_ticks(ax)
                
                # Draw Diagonal Line
                ax.plot([-99,99],[-99,99],ls='solid',lw=0.75,color="k")
                ax.set_ylim(vlims)
                ax.set_xlim(vlims)
                return fig,ax
                
                
                                      
                
                
            
            
            # ------ Make for 1 lag and 1 ensemble member ---------------------
            
            ilag             = 0
            
            
            vlims            = [-2,2]
            
            for ilag in range(12):
                fig,ax           = init_scatter(vlims=vlims)
                
                # Compute the lagcovariance (with detrending)
                datain_base      = varsin_ens[0][:,:].T # transpose to [month x year]
                datain_lag       = varsin_ens[0][:,:].T # transpose to [month x year]
                
                
                
                varbaseall = datain_base[im,:]
                varlagall  = datain_lag[im+ilag,:]
                
                th               = 0
                for th in range(3):
                    yr_mask          = np.where(data_mon_classes == th)[0] # Indices of valid years
                    varbase          = datain_base[im,yr_mask]
                    varlag           = datain_lag[(im+ilag)%12,yr_mask]
                    if detrend_anom:
                        varbase  = varbase - varbase.mean()
                        varlag   = varlag  - varlag.mean()
                    ax.scatter(varbase,varlag,alpha=0.75,marker='d',c=threscols[th],
                               label=r"Thres %i, $\rho=%.2f$ (n=%i)" % (th,stats.pearsonr(varbase,varlag)[0],varbase.shape[0]))
                
    
                
                
                # Plot Thresholds
                ax.axvline([thresset[0]],ls='dotted',color="b",lw=0.75,)
                ax.axvline([thresset[1]],ls='dotted',color="r",lw=0.75,)
                
                
                #  Plot the Lag Scatter
                #fig,ax = plt.subplots(1,1)
                
                
                ax.scatter(varbaseall,varlagall,s=25,alpha=0.55,color='k',zorder=-3,marker='x')
                ax.set_title("Ens %02i, Base Month: %s (Lag %0i)\nTotal Correlation = %.2f" % (e+1,mons3[im],ilag,stats.pearsonr(varbaseall,varlagall)[0]))
                ax.legend()
                
                
                savename = "%sScatterPlotCorr_%s_Ens%02i_Basemon%02i__dtanom%i_Lag%02i.png" % (figpath,locfn,e+1,im+1,detrend_anom,ilag)
                plt.savefig(savename,dpi=150,bbox_inches="tight")
            
            # ------- -------
        
        
        for th in range(nthres):
            if th < nthres-1:
                print("th=%i, Doing a specific threshold where c=%s" % (th,th))
            
                yr_mask          = np.where(data_mon_classes == th)[0] # Indices of valid years
                
                # Compute the lagcovariance (with detrending)
                datain_base      = varsin_ens[0][:,:].T # transpose to [month x year]
                datain_lag       = varsin_ens[0][:,:].T # transpose to [month x year]
                
                ac,yr_count      = proc.calc_lagcovar(datain_base,datain_lag,lags,im+1,0,yr_mask=yr_mask,debug=False)
                
                acs_all[e,im,th,:]  = ac.copy()
                counts_all[e,im,th] = yr_count
            else: #Don't apply a year masks
                print("th=%i,Doing all" % th)
                # Compute the lagcovariance (with detrending)
                datain_base      = varsin_ens[0][:,:].T # transpose to [month x year]
                datain_lag       = varsin_ens[0][:,:].T # transpose to [month x year]
                
                ac      = proc.calc_lagcovar(datain_base,datain_lag,lags,im+1,0,debug=False)
                
                acs_all[e,im,th,:]  = ac.copy()
                counts_all[e,im,th] = yr_count
            
    
    #proc.class_count(data_mon_classes)

    
#%% Save the output above
threslabs = ["x < -1","-1 < x <= 1","x > 1", "ALL"]
coords    = dict(ens=np.arange(1,43,1),mon=np.arange(1,13,1),thresholds=threslabs,lag=lags)
coords_c  = dict(ens=np.arange(1,43,1),mon=np.arange(1,13,1),thresholds=threslabs)
da_acs    = xr.DataArray(acs_all,coords=coords,dims=coords,name="acfs")
da_counts = xr.DataArray(counts_all,coords=coords_c,dims=coords_c,name="counts")
da_out    = xr.merge([da_acs,da_counts])
edict     = proc.make_encoding_dict(da_out)


#savename  = "%sCESM1_HTR_SST_SSS_CrossCorr_%s.nc" % (outpath,locfn)
#da_out.to_netcdf(savename,encoding=edict,)

#%% Plot ACF

im         = 1
xtks       = np.arange(0,63,3)

thresnames = ["Negative","Neutral","Positive","ALL"]
threscols  = ["royalblue","gray","orangered","black"]

fig,ax     = plt.subplots(1,1,constrained_layout=True,figsize=(10,4))
ax,_       = viz.init_acplot(im,xtks,lags)

for th in range(4):
    
    # Plot Ensemble Members
    for e in range(nens):
        plotvar = da_acs.isel(mon=im,ens=e,thresholds=th)
        ax.plot(lags,plotvar,c=threscols[th],alpha=0.05,label="",zorder=-3)
        
    
    # Plot Ensemble Average
    mu    = da_acs.isel(mon=im,thresholds=th).mean('ens')
    sigma = da_acs.isel(mon=im,thresholds=th).std('ens')
    ax.plot(lags,mu,c=threscols[th],label=thresnames[th])
    ax.fill_between(lags,mu-sigma,mu+sigma,label="",color=threscols[th],alpha=0.1,zorder=-2)
ax.legend(ncols=4)

ax.axhline([0],ls='dashed',lw=0.55,color="k")


savename = "%sPosNeg_ACF_SST_%s_mon%02i.png" % (figpath,locfn,im+1)

plt.savefig(savename,dpi=150,bbox_inches='tight')
#for e in range(nens):


#%% Instead of doing "Loop by mon", do things from here


im = 1



varsin_ens = [v[:,:,:,:].squeeze() for v in varsin]  # [42,86,12]
varsmon    = [v[:,:,:,im].squeeze() for v in varsin] # (42,86)

# The samespl for that month (42 * 86)
monsamples       = varsmon[0].flatten()

allsamples       = varsin_ens[0].reshape(nens*nyrs,12)

# Classify the samples
data_mon_classes = proc.make_classes_nd(monsamples,thresholds,dim=0,debug=False).squeeze() # Use the Base Variable


#%% Copute the lag correlation

lags  = np.arange(37)
nlags = len(lags)
dropexceed = True


lagcorrs    = np.zeros((nlags,nthres)) * np.nan
threscounts = np.zoers((nlags,nthres))
for th in range(3):
    
    # Get Indices for events
    id_class        = np.where(data_mon_classes==th)[0]
    
    
    # Get Input
    x1 = monsamples[id_class]
    
    for lag in range(nlags):
        
        # Get output
        # idclass_ens,idclass_yr = np.unravel_index(id_class,(nens,nyrs))
        # x2 = varsin_ens[idclass_ens[0].squeeze(),idclass_yr[0].squeeze(),im+lag]
        imlag = im+lag
        if imlag > 11:
            # First compute year shift
            yrshift = int(imlag/12)
            idclass_ens,idclass_yr = np.unravel_index(id_class,(nens,nyrs))
            idclass_ens = idclass_ens.squeeze()
            idclass_yr  = idclass_yr.squeeze() + yrshift # Shift year forward
            
            idexceed = np.where(idclass_yr >= nyrs)[0]
            if dropexceed: # Drop the Data
                idclass_ens = np.delete(idclass_ens,idexceed)
                idclass_yr  = np.delete(idclass_yr,idexceed)
            else:
                idclass_yr[idexceed] = 0 # Shift it back to the front
            
            # Two Approaches (these are equivalent!)
            
            # 1) ravel multi index
            hey     = np.ravel_multi_index((idclass_ens,idclass_yr),(nens,nyrs))
            x1_new2 = monsamples[hey]
            
            # 2) Manuall do this thru reshape
            monsamples_unravel = monsamples.reshape(nens,nyrs)
            x1_new             = x111[idclass_ens,idclass_yr]
            

            
            
            
            
            
            # Mod 12
            imlag = imlag % 12
        else:
            
            
            x1                  = monsamples[id_class]
            x2                  = allsamples[id_class,imlag]
        
        
        threscounts[lag,th] = x1.shape[0]
        
        # Compute Correlation
        corr_out = np.corrcoef(x1,x2)[0,1]
        
        # Append
        lagcorrs[lag,th] = corr_out
    


#%%
    


# #%% 
# for th in range(nthres): # Loop for each threshold
    
#     if th < nthres + 1: # Calculate/Loop for all points
#         print("th=%i, doing threshold %i" % (th,thresholds[th]))
#     else:
#         print("th=%i, doin for all"% th)




# #%% Prepare for input
# # -------------------

# def make_mask(var_in,sumdims=[0,]):
#     vsum                  = np.sum(var_in,sumdims)
#     vsum[~np.isnan(vsum)] = 1
#     return vsum

# # Indicate inputs
# dsin                 = [ds_base,ds_lag]
# vnames_in            = [vname_base,vname_lag] 

# # Transpose and read out files  (Make into [lon x lat x ens x time])
# varsin               = [dsin[vv][vnames_in[vv]].transpose('lon','lat','ensemble','time').values for vv in range(len(dsin))]

# # Get Dimensions
# nlon,nlat,nens,ntime = varsin[0].shape
# npts                 = nlon*nlat
# varsin               = [v.reshape(npts,nens,ntime) for v in varsin]

# # Make sure they have consistent masks
# vmasks               = [make_mask(vin,sumdims=(2,)) for vin in varsin]
# maskfin              = np.prod(np.array(vmasks),0) # Make Product of masks (if nan in one, nan in all...) # [np.nan]
# varsin               = [v * maskfin[:,:,None] for v in varsin]

# # Get Threshold Informaton
# nthres               = len(thresholds) + 1  + 1 # less than, between, greater, and ALL
# nlags                = len(lags)




# #var_base,var_lag     = varsin

    

# # Repeat for thresholding variable, if option is set
# if thresvar is True:
#     print("WARNING NOT IMPLEMENTED. See Old Script...")
    
    
# # ----------------------
# #%% Perform calculations
# # ----------------------

# """
# Inputs are:
#     1) variable [ens x time x lat x lon]
#     2) lon      [lon]
#     3) lat      [lat]
#     4) thresholds [Numeric] (Standard Deviations)
#     5) savename [str] Full path to output file
#     6) loadvar(optional) [lon x lat x time x otherdims] (thresholding variable)
    
# """

# for e in range(nens):
    
    
    
#     # Remove NaN Points
#     ensvars   = [invar[:,e,:] for invar in varsin] # npts x time
#     nandicts  = [proc.find_nan(ensv,1,return_dict=True,verbose=False) for ensv in ensvars]
#     validdata = [nd['cleaned_data'] for nd in nandicts] # [pts x yr x mon]
    
#     nptsvalid = [vd.shape[0] for vd in validdata]
#     if np.all([i == nptsvalid[0] for i in nptsvalid]):
#         npts_valid = nptsvalid[0]
#     else:
#         print("WARNING, NaN points are not the same across variables. Aborting loop...")
#         npts_valid = np.nan
#         break
    
#     # Split to year and month
#     nyr        = int(ntime/12)
#     validdata  = [vd.reshape(npts_valid,nyr,12) for vd in validdata]

    
#     # Preallocate
#     class_count = np.zeros((npts_valid,12,nthres)) # [pt x eventmonth x threshold]
#     sst_acs     = np.zeros((npts_valid,12,nthres,nlags))  # [pt x eventmonth x threshold x lag]
#     #sst_cfs     = np.zeros((npts_valid,12,nthres+2,nlags,2))  # [pt x eventmonth x threshold x lag x bounds]
    
    
#     for im in range(12):
        
#         # For that month, determine which years fall into which thresholds [pts,years]
#         data_mon = [vd[:,:,im] for vd in validdata] # [pts x yr]
        
#         if thresvar:
#             print("WARNING NOT IMPLEMENTED. See Old Script...")
#         else:
#             data_mon_classes = proc.make_classes_nd(data_mon[0],thresholds,dim=1,debug=False) # Use the Base Variable
            
#         for th in range(nthres): # Loop for each threshold
            
#             if th < nthres + 1: # Calculate/Loop for all points
#                 #print(th)
#                 for pt in tqdm(range(npts_valid)): 
                    
#                     # Get years which fulfill criteria
#                     yr_mask          = np.where(data_mon_classes[pt,:] == th)[0] # Indices of valid years
#                     if len(yr_mask) < 2:
#                         print("Only 1 point found for pt=%i, th=%i" % (pt,th))
#                         continue
                    
#                     # Compute the lagcovariance (with detrending)
#                     datain_base      = validdata[0][pt,:,:].T # transpose to [month x year]
#                     datain_lag       = validdata[1][pt,:,:].T # transpose to [month x year]
                    
#                     ac,yr_count      = proc.calc_lagcovar(datain_base,datain_lag,lags,im+1,0,yr_mask=yr_mask,debug=False)
#                     #cf = proc.calc_conflag(ac,conf,tails,len(yr_mask)) # [lags, cf]
                    
#                     # Save to larger variable
#                     class_count[pt,im,th] = yr_count
#                     sst_acs[pt,im,th,:]   = ac.copy()
#                     #sst_cfs[pt,im,th,:,:]  = cf.copy()
#                     # End Loop Point -----------------------------
            
#             else: # Use all Data
#                 #print("Now computing for all data on loop %i"%th)
#                 # Reshape to [month x yr x npts]
#                 datain_base    = validdata[0].transpose(2,1,0)
#                 datain_lag     = validdata[1].transpose(2,1,0)
#                 acs            = proc.calc_lagcovar_nd(datain_base,datain_lag,lags,im+1,0) # [lag, npts]
#                 #cfs = proc.calc_conflag(acs,conf,tails,nyr) # [lag x conf x npts]
                
#                 # Save to larger variable
#                 sst_acs[:,im,th,:] = acs.T.copy()
#                 #sst_cfs[:,im,th,:,:]  = cfs.transpose(2,0,1).copy()
#                 class_count[:,im,th]   = nyr
#             # End Loop Threshold -----------------------------
#         # End Loop Event Month -----------------------------
    
    
#     #% Now Replace into original matrices
#     # Preallocate
    
#     count_final = np.zeros((npts,12,nthres)) * np.nan
#     acs_final   = np.zeros((npts,12,nthres,nlags)) * np.nan
#     #cfs_final   = np.zeros((npts,12,nthres+2,nlags,2)) * np.nan
    
    
#     # Replace
#     okpts_var                  = nandicts[0]['ok_indices'] # Basevar
#     count_final[okpts_var,...] = class_count
#     acs_final[okpts_var,...]   = sst_acs
#     #cfs_final[okpts,...]  = sst_cfs
    
#     # Reshape
#     count_final = count_final.reshape(nlon,nlat,12,nthres)
#     acs_final   = acs_final.reshape(nlon,nlat,12,nthres,nlags)
    
    
#     # Get Threshold Labels
#     threslabs   = []
#     if nthres == 1:
#         threslabs.append("$T'$ <= %i"% thresholds[0])
#         threslabs.append("$T'$ > %i" % thresholds[0])
#     elif nthres == 4:
#         for th in range(nthres-1):
#             if th == 0:
#                 tstr = "x < %s" % thresholds[th]
#             elif th == 1:
#                 tstr = "%s < x =< %s" % (thresholds[0],thresholds[1])
#             elif th == 2:
#                 tstr = "x > %s" % (thresholds[1])
#             threslabs.append(tstr)
#     else:
#         threslabs = [th for th in range(nthres-1)]
#     threslabs.append("ALL")
    
    
#     # Make into Dataset
#     coords_count = {'lon':lon,
#                     'lat':lat,
#                     'mons':np.arange(1,13,1),
#                     'thres':threslabs}
    
#     coords_acf  = {'lon'    :lon,
#                     'lat'   :lat,
#                     'mons'  :np.arange(1,13,1),
#                     'thres' :threslabs,
#                     'lags'  :lags}
    
#     da_count   = xr.DataArray(count_final,coords=coords_count,dims=coords_count,name="class_count")
#     da_acf     = xr.DataArray(acs_final,coords=coords_acf,dims=coords_acf,name="acf")
#     ds_out     = xr.merge([da_count,da_acf])
#     encodedict = proc.make_encoding_dict(ds_out)
    
#     # Save Output
#     savename = "%s%s_%s_ens%02i.nc" % (outpath,outname_data,lagname,e+1)
#     ds_out.to_netcdf(savename,encoding=encodedict)


#%%

#%% Do the calculations

#print("Script ran in %.2fs!"%(time.time()-st))
#print("Output saved to %s."% (savename))