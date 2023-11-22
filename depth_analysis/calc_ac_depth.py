#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Get UOHC Data for a single point


Created on Fri Jun 10 13:35:47 2022

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
stormtrack = 1 # Set to True to run on stormtrack, False for local run

# Data Preprocessing
# ------------------
startyr     = 1920
endyr       = 2006


# SPG Test Point
lonf        = -30+360
latf        = 50

# SPG Center
# lonf        = -40+360
# latf        = 53

# Transition Zone
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
varname    = "SALT" # ["SALT","TEMP"]..["TS","SSS","SST]


# MLD DAta
outpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/"
mldname = "CESM1_PiC_HMXL_Clim_Stdev.nc" # Made with viz_mldvar.py (stochmod/analysis)


# Set to False to not apply a mask (otherwise specify path to mask)
loadmask   = False #"/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/limask180_FULL-HTR.npy"
glonpath   = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/CESM1_lon180.npy"
glatpath   = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/CESM1_lat.npy"

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
    if varname == "TEMP":
        datpath = "/stormtrack/data4/share/deep_learning/data_yuchiaol/cesm_le/TEMP/"
    elif varname == "SALT":
        datpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/00_Commons/CESM1_LE/SALT/"
    
    # Output Paths
    figpath = "/stormtrack/data3/glliu/02_Figures/20220622/"
    outpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/CESM1/"
    
else:
    # Module Paths
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")

    # Input Paths 
    datpath = ""
    
    # Output Paths
    figpath     = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20220325/'
    outpath     = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/'
    
# Import modules
from amv import proc,viz
import scm


def find_tlatlon(ds,lonf,latf):

    # Get minimum index of flattened array
    kmin      = np.argmin( (np.abs(ds.TLONG-lonf) + np.abs(ds.TLAT-latf)).values)
    klat,klon = np.unravel_index(kmin,ds.TLAT.shape)
    
    # Print found coordinates
    foundlat = ds.TLAT.isel(nlat=klat,nlon=klon).values
    foundlon = ds.TLONG.isel(nlat=klat,nlon=klon).values
    print("Closest lon to %.2f was %.2f" % (lonf,foundlon))
    print("Closest lat to %.2f was %.2f" % (latf,foundlat))
    
    return ds.isel(nlon=klon,nlat=klat)
    
# -----------------------
#%% Get MLD Data
# -----------------------

# Get Mean Climatological Cycle
# Get Stdev

dsmld = xr.open_dataset(outpath+mldname)

if lonf > 180:
    lonfw = lonf-360
else:
    lonfw = lonf
    
# Get Mean and Stdev, take maximum of that
hbar     = dsmld.clim_mean.sel(lon=lonfw,lat=latf,method='nearest').max().values
mmax     = dsmld.clim_mean.sel(lon=lonfw,lat=latf,method='nearest').values.argmax()
hstd     = dsmld.stdev.sel(lon=lonfw,lat=latf,method='nearest').isel(month=mmax).values

# Convert to CM
hmax_sel =(hbar+hstd)*100

# -----------------------
#%% Load Data
# -----------------------
# Get the list of files
ncsearch = "b.e11.B20TRC5CNBDRD.f09_g16.*.pop.h.%s.*.nc" % varname
nclist   = glob.glob(datpath+ncsearch)
nclist   = [nc for nc in nclist if "OIC" not in nc]
nclist.sort()
nens     = len(nclist)
print("Found %i files!"%nens)


# Open and Slice
dsall = [] # This is just a remnant, remove it at some point...
for n in tqdm(range(nens)):
    
    # Open Dataset
    nc = nclist[n]
    ds = xr.open_dataset(nc)
    
    # Drop unwanted variables
    varkeep = [varname,"TLONG","TLAT","z_t","time"]
    dsvars  = list(ds.variables)
    remvar  = [i for i in dsvars if i not in varkeep]
    ds      = ds.drop(remvar)
    
    # Slice to Time
    ds      = ds.sel(time=slice("%s-02-01"%(startyr),"%s-01-01"%(endyr+1)))
    
    # Select a Point
    ds      = proc.find_tlatlon(ds,lonf,latf,verbose=False)
    
    # Select a Depth
    ds      = ds.sel(z_t=slice(0,hmax_sel))
    
    # Save into an array
    if n == 0:
        v_all = np.zeros((nens,)+ds[varname].shape) * np.nan # [ens x time x depth]
        z_t   = ds.z_t.values
        times = ds.time.values
    v_all[n,:,:] = ds[varname].values
    
    # Append and move on...
    dsall.append(ds)


tlon = ds.TLONG.values
tlat = ds.TLAT.values

# Resave the variable as an array
dims = {"ensemble":np.arange(1,43,1),
          "time":times,
          "z_t":z_t
          }

attr_dict = {'lon':tlon,
             'lat':tlat,
             'hbar':hbar,
             'hstd':hstd,
             'hmax':hmax_sel,
             }

da = xr.DataArray(v_all,
    dims=dims,
    coords=dims,
    name = varname,
    attrs=attr_dict
    )

savename = "%sCESM1LE_UO%s_lon%i_lat%i.nc" % (outpath,varname,lonf,latf)
#% Save as netCDF
# ---------------
st = time.time()
encoding_dict = {varname : {'zlib': True}} 
print("Saving as " + savename)
da.to_netcdf(savename,
         encoding=encoding_dict)
print("Saved in %.2fs" % (time.time()-st))

#%% Load in data, preprocess, and compute the autocorrelation

# Load in the data
savename = "%sCESM1LE_UO%s_lon%i_lat%i.nc" % (outpath,varname,lonf,latf)
ds       = xr.open_dataset(savename)
T        = ds[varname].values
z        = ds.z_t.values
times    = ds.time.values
nens,ntime,nz = T.shape # [ens x time x depth]


# Remove the seasonal cycle (monthly anomalies)
nyrs   = int(ntime/12)
vbar,v = proc.calc_clim(T,1,returnts=1)
vprime = v - vbar[:,None,:,:] # [ens x yr x mon x depth]
vprime = vprime.reshape(nens,ntime,nz)


# Remove the ensemble average
vprime = vprime - vprime.mean(0)[None,...]

# Transpose to input dimensions
invar         = vprime.transpose(1,2,0) # [time x depth x ens]


# Old section (delete soon)
#.reshape(ntime,nens*nz)[None,None,:,:] # [1 x 1 x time x depth * ens]

#%% Do the calculations for autocorrelation (copied from pointwise_autocorrelation)
"""
Inputs are:
    1) variable [1 x 1 x time x otherdims]
    2) lon      [lon]
    3) lat      [lat]
    4) thresholds [Numeric] (Standard Deviations)
    5) savename [str] Full path to output file
    
"""

zref = 0 # Select Depth Reference Level

# First things first, combine lat/lon/otherdims, remove nan points
ntime,nz,nens = invar.shape

# # Get Dimensions
# if len(invar.shape) > 3:
    
#     print("%s has more than 3 dimensions. Combining." % varname)
#     nlon,nlat,ntime,notherdims = invar.shape
    
#     # Commented for this script, but might need to fix otherwise
#     invar = invar.transpose(0,1,3,2) # [nlon,nlat,otherdims,time]
#     npts = nlon*nlat*notherdims # combine ensemble and points
    
# else:
#     notherdims      = 0
#     nlon,nlat,ntime = invar.shape
#     npts            = nlon*nlat

nyr             = int(ntime/12)
nlags           = len(lags)
nthres          = len(thresholds)


# Remove Broken timestep
invarrs     = invar.copy() # [ntime, ndepth, nens]
if varname in ["SSS","TEMP","SALT"]:
    invarrs[219,:,:] = 0 # There is something wrong with this timestep, ocean?
npts_valid  = nens

# Split to year x month
invar_valid = invarrs.reshape(nyr,12,nz,nens)
invar_valid = invar_valid.transpose(2,3,0,1) # [depth x ens x yr x mon]

# Preallocate (nthres + 1 (for all thresholds), and last is all data)
class_count   = np.zeros((npts_valid,12,nz,nthres+2)) # [pt x eventmonth x depth x threshold]
invar_acs     = np.zeros((npts_valid,12,nz,nthres+2,nlags))  # [pt x eventmonth x depth x threshold x lag]
invar_cfs     = np.zeros((npts_valid,12,nz,nthres+2,nlags,2))  # [pt x eventmonth x depth x threshold x lag x bounds]

# A pretty ugly loop....
# Now loop for each month
for im in range(12):
    print(im)
    
    # For that month, determine which years fall into which thresholds [pts,years]
    invar_mon         = invar_valid[zref,:,:,im] # [pts x yr]
    invar_mon_classes = proc.make_classes_nd(invar_mon,thresholds,dim=1,debug=False)
    
    
    for kz in tqdm(range(nz)): # Loop for each depth
    
        for th in range(nthres+2): # Loop for each threshold
        
            if th < nthres + 1: # Calculate/Loop for all points
                for pt in range(npts_valid): 
                    
                    # Get years which fulfill criteria
                    yr_mask     = np.where(invar_mon_classes[pt,:] == th)[0] # Indices of valid years
                    
                    #invar_in      = invar_valid[pt,yr_mask,:] # [year,month]
                    #invar_in      = invar_in.T
                    #class_count[pt,im,th] = len(yr_mask) # Record # of events 
                    #ac = proc.calc_lagcovar(invar_in,invar_in,lags,im+1,0) # [lags]
                    
                    # Compute the lagcovariance (with detrending)
                    invar_base  = invar_valid[zref,pt,:,:].T # transpose to [month x year]
                    invar_targ  = invar_valid[kz,pt,:,:].T # 
                    
                    ac,yr_count = proc.calc_lagcovar(invar_base,invar_targ,lags,im+1,0,yr_mask=yr_mask,debug=False)
                    cf          = proc.calc_conflag(ac,conf,tails,len(yr_mask)) # [lags, cf]
                    
                    # Save to larger variable
                    class_count[pt,im,kz,th]    = yr_count
                    invar_acs[pt,im,kz,th,:]    = ac.copy()
                    invar_cfs[pt,im,kz,th,:,:]  = cf.copy()
                    # End Loop Point -----------------------------
            
            else: # Use all Data
                #print("Now computing for all data on loop %i"%th)
                # Reshape to [month x yr x npts]
                invar_base    = invar_valid[zref,:,:,:].transpose(2,1,0)
                invar_targ    = invar_valid[kz,:,:,:].transpose(2,1,0)
                
                acs = proc.calc_lagcovar_nd(invar_base,invar_targ,lags,im+1,1) # [lag, npts]
                cfs = proc.calc_conflag(acs,conf,tails,nyr) # [lag x conf x npts]
                
                # Save to larger variable
                invar_acs[:,im,kz,th,:]    = acs.T.copy()
                invar_cfs[:,im,kz,th,:,:]  = cfs.transpose(2,0,1).copy()
                class_count[:,im,kz,th]    = nyr
            # End Loop Threshold -----------------------------
        # End Loop Depth
    # End Loop Event Month -----------------------------

#% Now Replace into original matrices
# Preallocate
count_final = class_count   #np.zeros((nens,12,nz,nthres+2)) * np.nan
acs_final   = invar_acs   #np.zeros((nens,12,nz,nthres+2,nlags)) * np.nan
cfs_final   = invar_cfs #np.zeros((nens,12,nz,nthres+2,nlags,2)) * np.nan

# # Replace
# count_final[okpts,...] = class_count
# acs_final[okpts,...]   = invar_acs
# cfs_final[okpts,...]   = invar_cfs

# # Reshape output
# if notherdims == 0:
#     count_final = count_final.reshape(nlon,nlat,12,nthres+2)
#     acs_final   = acs_final.reshape(nlon,nlat,12,nthres+2,nlags)
#     cfs_final   = cfs_final.reshape(nlon,nlat,12,nthres+2,nlags,2)
# else:
#     count_final = count_final.reshape(nlon,nlat,notherdims,12,nthres+2)
#     acs_final   = acs_final.reshape(nlon,nlat,notherdims,12,nthres+2,nlags)
#     cfs_final   = cfs_final.reshape(nlon,nlat,notherdims,12,nthres+2,nlags,2)

# Get variable symbol
if varname in ['TS','SST',"TEMP"]:
    vsym = "T"
elif varname in ["SSS","SALT"]:
    vsym = "S"

# Get Threshold Labels
threslabs   = []
if nthres == 1:
    threslabs.append("$%s'$ <= %i"% (vsym,thresholds[0]))
    threslabs.append("$%s'$ > %i" % (vsym,thresholds[0]))
else:
    for th in range(nthres):
        thval= thresholds[th]
        
        if thval != 0:
            sig = ""
        else:
            sig = "$\sigma$"
        
        if th == 0:
            tstr = "$%s'$ <= %i %s" % (vsym,thval,sig)
        elif th == nthres:
            tstr = "$%s'$ > %i %s" % (vsym,thval,sig)
        else:
            tstr = "%i < $%s'$ =< %i %s" % (vsym,thresholds[th-1],thval,sig)
        threslabs.append(th)
threslabs.append("ALL")


#%% Unique to Depth v. Lag Analysis, Separate Dimensions and Save.

# Reshape variables (separate depth v lag)
#acs_final   = acs_final.squeeze().reshape(nens,nz,12,nthres+2,nlags,) # [ens x depth x month x thres x lag]
#cfs_final   = cfs_final.squeeze().reshape(nens,nz,12,nthres+2,nlags,2)
#count_final = count_final.squeeze().reshape(nens,nz,12,nthres+2)
#count_final = count_final.squeeze().reshape(12,)

# Debugging Plot
fig,ax = plt.subplots(1,1)
for i in range(42):
    ax.plot(lags,acs_final[i,0,1,-1,:])

# savename = "%sCESM1LE_UOTEMP_lon%i_lat%i.nc" % (outpath,tlon,tlat)

savename = "%s%s_Autocorrelation_DepthvLag_lon%i_lat%i_%s.npz" % (outpath,varname,lonf,latf,lagname)

#% Save Output
np.savez(savename,**{
    'class_count' : count_final,
    'acs' : acs_final,
    'cfs' : cfs_final,
    'thresholds' : thresholds,
    'lon' : tlon,
    'lat' : tlat,
    'lags': lags,
    'threslabs' : threslabs,
    "z_t" : z,
    },allow_pickle=True)

print("Script ran in %.2fs!"%(time.time()-st))
print("Output saved to %s."% (savename))
